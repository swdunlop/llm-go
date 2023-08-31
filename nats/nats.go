// Package nats registers an llm implementation that uses NATS to communicate with a worker process.  This is useful
// for building a multi-tier system where the workers are on isolated machines dedicated to LLM.
package nats

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nuid"
	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/llm-go/internal/slog"
	msg "github.com/swdunlop/llm-go/nats/protocol"
)

func init() {
	llm.Register("nats", func(cf map[string]string) (llm.Interface, error) {
		opts := Defaults()
		err := llm.Unmap(cf, &opts)
		if err != nil {
			return nil, err
		}
		return NewNATS(nil, opts)
	})
}

// NewNATS creates a new NATS client that supports prediction using the provided NATS connection and configuration.
// The configuration should specify `nats_worker_subject` to identify the worker that will be used to perform the
// request, otherwise llm.worker.default will be used.
func NewNATS(conn *nats.Conn, cf Options) (*Client, error) {
	var err error
	ct := new(Client)
	ct.options = cf
	ct.conn = conn
	if ct.conn == nil {
		ct.conn, err = ct.options.Dial(
			nats.ErrorHandler(ct.handleNatsError),
		)
		if err != nil {
			return nil, err
		}
		ct.release = ct.conn.Close
	}
	ct.nuid = nuid.New()
	ct.stream.done = make(chan struct{})
	ct.stream.inbox = nats.NewInbox()
	ct.stream.ch = make(chan *nats.Msg, 16) // give NATS some slack.
	ct.stream.subscription, err = ct.conn.ChanSubscribe(ct.stream.inbox, ct.stream.ch)
	if err != nil {
		if ct.release != nil {
			ct.release()
		}
		return nil, err
	}
	ct.stream.subscribers = make(map[string]chan<- *msg.WorkerResponse)
	go ct.processStreamResponses()
	return ct, nil
}

type Client struct {
	options Options

	conn    *nats.Conn
	cancel  func() // cancels the stream handler.
	release func() // used if NewNATS opened the connection
	stream  struct {
		subscription *nats.Subscription
		inbox        string         // used for stream responses
		ch           chan *nats.Msg // used for stream responses, closed when Release is called.
		done         chan struct{}  // closed when the stream handler exits.
		control      sync.Mutex
		subscribers  map[string]chan<- *msg.WorkerResponse
	}
	nuid *nuid.NUID
}

func (ct *Client) handleNatsError(conn *nats.Conn, sub *nats.Subscription, err error) {
	switch err {
	case nats.ErrSlowConsumer:
		pendingMsgs, _, err := sub.Pending()
		if err == nil {
			slog.Warn(`nats slow consumer`, `subject`, sub.Subject, `pending`, pendingMsgs)
			return
		}
	}
	slog.Error(`nats error`, `error`, err, `subject`, sub.Subject)
}

// Predict implements the llm.Predictor interface by sending a request to the worker and waiting for a response.
func (ct *Client) Predict(ctx context.Context, options map[string]string, input []string, fn func(llm.Prediction) error) (output string, err error) {
	return ct.PredictNATS(ctx, options, input, fn)
}

// PredictNATS is like the standard predict interface but can convey additional options to the worker.
func (ct *Client) PredictNATS(ctx context.Context, options map[string]string, input []string, fn func(llm.Prediction) error) (output string, err error) {
	job := ct.nuid.Next()
	stream := ""
	if fn != nil {
		stream = ct.stream.inbox
	}
	req := msg.WorkerRequest{Job: job, Predict: &msg.PredictRequest{
		Input:   input,
		Stream:  stream,
		Options: options,
	}}
	data, err := json.Marshal(&req)
	if err != nil {
		return "", err
	}
	slog.From(ctx).Debug(`sending request`, `subject`, ct.options.WorkerSubject, `stream`, stream)
	nm, err := ct.conn.RequestWithContext(ctx, ct.options.WorkerSubject, data)
	if err != nil {
		return "", err
	}
	var ret msg.WorkerResponse
	err = json.Unmarshal(nm.Data, &ret)
	if err != nil {
		return "", err
	}
	if ret.Error != nil {
		return "", ret.Error
	}
	if ret.Predict != nil {
		return ret.Predict.Output, nil
	}
	if fn == nil {
		return "", fmt.Errorf(`empty response from worker`) // should only happen when streaming.
	}

	streamCh := make(chan *msg.WorkerResponse, 16) // TODO: why do we need so much buffer?
	ct.subscribeJob(job, streamCh)
	defer ct.unsubscribeJob(job, streamCh)
	for {
		select {
		case <-ctx.Done():
			ct.interrupt(job)
			return "", ctx.Err()
		case msg := <-streamCh:
			ok := false
			if msg == nil {
				return "", fmt.Errorf(`stream closed`)
			}
			if msg.Error != nil {
				return "", msg.Error
			}
			if msg.Stream != nil {
				err = fn(msg.Stream)
				if err != nil {
					return "", err
				}
				ok = true
			}
			if msg.Predict != nil {
				return msg.Predict.Output, nil
			}
			if !ok {
				slog.From(ctx).Warn(`unexpected response`, `response`, msg)
			}
		}
	}
}

func (ct *Client) subscribeJob(job string, ch chan *msg.WorkerResponse) {
	ct.stream.control.Lock()
	ct.stream.subscribers[job] = ch
	ct.stream.control.Unlock()
}

func (ct *Client) unsubscribeJob(job string, ch chan *msg.WorkerResponse) {
	ct.stream.control.Lock()
	delete(ct.stream.subscribers, job)
	ct.stream.control.Unlock()
}

func (ct *Client) processStreamResponses() {
	defer func() {
		defer close(ct.stream.done)
		ct.stream.control.Lock()
		defer ct.stream.control.Unlock()
		for _, ch := range ct.stream.subscribers {
			close(ch) // TODO: ensure the sessions understand that the channel is closed.
		}
	}()

	for msg := range ct.stream.ch {
		ct.processStreamData(msg.Data)
	}
}

func (ct *Client) processStreamData(data []byte) {
	var ret msg.WorkerResponse
	err := json.Unmarshal(data, &ret)
	if err != nil {
		slog.Warn(`failed to unmarshal response`, `error`, err)
		return
	}
	job := ret.Job
	ct.stream.control.Lock()
	ch, ok := ct.stream.subscribers[job]
	ct.stream.control.Unlock()
	if !ok {
		slog.Warn(`received response for unknown job`, `job`, job)
		return
	}
	select {
	case ch <- &ret:
	default:
		slog.Warn(`dropping stream response`, `job`, job)
	}
}

func (ct *Client) interrupt(job string) {
	req := msg.WorkerRequest{Job: job, Interrupt: &msg.InterruptRequest{}}
	data, err := json.Marshal(&req)
	if err != nil {
		panic(err)
	}
	_, err = ct.conn.Request(ct.options.WorkerSubject, data, time.Second)
	if err != nil {
		slog.Warn(`failed to interrupt job`, `job`, job, `error`, err)
	}
}

// Release implements llm.Interface by unsubscribing from NATS.  Release will close the NATS connection if NewNATS
// opened it.  (This happens when NewNATS is called with a nil connection, or llm.New(`nats`) is used.)
func (ct *Client) Release() {
	if ct.stream.subscription != nil {
		err := ct.stream.subscription.Unsubscribe()
		if err != nil {
			slog.Error(`stream unsubscribe failed`, `error`, err, `subject`, ct.stream.inbox)
		}
	}
	if ct.stream.ch != nil {
		close(ct.stream.ch)
	}
	if ct.stream.done != nil {
		<-ct.stream.done
	}
	if ct.release != nil {
		ct.release()
	}
	ct.conn = nil
}

// Defaults returns the default options for a NATS client.
func Defaults() Options {
	return Options{
		URL:           nats.DefaultURL,
		WorkerSubject: `llm.worker.default`,
		ClientName:    `llm-client`,
	}
}

// Options describes the options used to create a NATS client.  This is unmarshalled from the configuration
// provided to llm.New or NewNATS.
type Options struct {
	// WorkerSubject identifies the NATS subject where requests should be sent.  This defaults to `llm.worker.default`,
	// which matches the worker's default WorkerSubject.
	WorkerSubject string `env:"nats_worker_subject"`

	// ClientName gives the client a name that is used to identify it in NATS server logs.  This defaults to
	// `llm-client`.
	ClientName string `env:"nats_client_name"`

	// URL specifies the NATS client URL used to connect to the NATS server.  This is used if Conn is nil, which is
	// the case if `llm.New("nats")` is used to create the client.  This defaults to `nats://localhost:4222`.
	URL string `env:"nats_url"`

	// NKeyFile provides the path to an nkey seed / secret file that will be used to authenticate with the NATS server.
	// Ignored if a NATS connection is provided.
	NKeyFile string `env:"nats_nk"`

	// CA provides the path to a file containing trusted certificates for verifying the NATS server.  Ignored if a
	// NATS connection is provided.  If not provided, the host certificate authorities will be used.
	CA string `env:"nats_ca"`
}

// Dial will connect to the NATS server using the options provided.
func (opts *Options) Dial(more ...nats.Option) (*nats.Conn, error) {
	options := make([]nats.Option, 0, 8)
	if opts.NKeyFile != `` {
		opt, err := nats.NkeyOptionFromSeed(opts.NKeyFile)
		if err != nil {
			return nil, err
		}
		options = append(options, opt)
	}
	if opts.CA != `` {
		options = append(options, nats.RootCAs(opts.CA))
	}
	if opts.ClientName != `` {
		options = append(options, nats.Name(opts.ClientName))
	}
	options = append(options, more...)
	return nats.Connect(opts.URL, options...)
}
