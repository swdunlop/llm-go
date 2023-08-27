// Package worker implements a NATS-based worker for large language models.
package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/nats-io/nats.go"
	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/llm-go/configuration"
	"github.com/swdunlop/llm-go/internal/slog"
	msg "github.com/swdunlop/llm-go/nats/protocol"
)

// Run will run a NATS-based worker with the provided NATS connection and configuration.
func Run(ctx context.Context, cf configuration.Interface, options ...Option) error {
	slog.From(ctx).Debug(`starting worker`)
	var w worker
	w.options = Options{
		LLMType:       "",
		NATSUrl:       nats.DefaultURL,
		WorkerSubject: "llm.worker.default",
		ClientName:    "llm-worker",
	}
	err := configuration.Unmarshal(&w.options, cf)
	if err != nil {
		return err
	}
	for _, opt := range options {
		opt(&w)
		if w.err != nil {
			return err
		}
	}

	if w.conn == nil {
		var err error
		w.conn, err = nats.Connect(w.options.NATSUrl, nats.Name(w.options.ClientName))
		if err != nil {
			return err
		}
		defer w.conn.Close()
	}

	ch := make(chan *nats.Msg)
	slog.From(ctx).Debug(`subscribing to worker subject`, `subject`, w.options.WorkerSubject)
	sub, err := w.conn.ChanQueueSubscribe(w.options.WorkerSubject, "llm-worker", ch)
	if err != nil {
		return err
	}

	unsubscribed := make(chan struct{})
	defer func() { <-unsubscribed }()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		defer close(unsubscribed)
		<-ctx.Done()
		err := sub.Unsubscribe()
		if err != nil {
			slog.Warn(`failed to unsubscribe from worker subject`, `subject`, w.options.WorkerSubject, `err`, err.Error())
		}
		close(ch)
	}()

	if w.model == nil {
		if w.options.LLMType == "" {
			return errors.New(`llm type must be specified`)
		}
		m, err := llm.New(w.options.LLMType, cf)
		if err != nil {
			return err
		}
		if p, ok := m.(llm.Predictor); !ok {
			return fmt.Errorf(`%q does not implement llm.Predictor`, w.options.LLMType)
		} else {
			w.model = p
		}
		defer w.model.Release()
	}

	w.job.ch = make(chan predictRequest, 4)
	go w.processPredictRequests()
	w.run(ctx, ch)
	return nil
}

// Options describes the configuration options for a NATS-based worker.  Note that this configuration will be passed on
// to the LLM driver, so the driver may have additional configuration options.
type Options struct {
	// LLM implementation to use, this must be specified and must implement llm.Predictor.
	LLMType string `cfg:"llm_type"`

	// NATSUrl is the URL of the NATS server to connect to, defaults to nats.DefaultURL.
	NATSUrl string `cfg:"nats_url"`

	// WorkerSubject is the NATS subject to subscribe to, defaults to llm.worker.default.
	WorkerSubject string `cfg:"worker_subject"`

	// ClientName gives the client a name that is used to identify it in NATS server logs.  This defaults to
	// "llm-worker".
	ClientName string `cfg:"nats_client_name"`
}

type worker struct {
	options Options
	hooks   []func(ctx context.Context, model llm.Predictor, req *msg.PredictRequest) error
	conn    *nats.Conn
	model   llm.Predictor
	job     struct {
		ch   chan predictRequest
		done chan struct{}

		control sync.RWMutex
		id      string
		cancel  func() // cancels a prediction
	}
	err error // used by hooks to indicate a fatal error.
}

func (w *worker) run(ctx context.Context, ch chan *nats.Msg) {
	defer func() {
		close(w.job.ch)
		<-w.job.done
	}()
	for msg := range ch {
		w.process(ctx, msg)
	}
}

// process will only return an error for fatal errors.
func (w *worker) process(ctx context.Context, nm *nats.Msg) {
	var req msg.WorkerRequest
	err := json.Unmarshal(nm.Data, &req)
	switch {
	case err != nil:
		w.reject(ctx, nm.Reply, msg.ErrIllegibleRequest, err.Error())
	case req.Job == "":
		w.reject(ctx, nm.Reply, msg.ErrInvalidRequest, "job id is required")
		return
	}

	os.Stdout.Write(nm.Data)

	ok := false
	if req.Interrupt != nil {
		w.interrupt(ctx, nm.Reply, req.Job)
		ok = true
	}
	if req.Predict != nil {
		w.predict(ctx, nm.Reply, req.Job, req.Predict)
		ok = true
	}

	if !ok {
		w.reject(ctx, nm.Reply, msg.ErrUnsupportedCommand, "command not supported")
	}
}

// predict must be done out of process to avoid blocking the NATS worker.
func (w *worker) predict(ctx context.Context, reply, job string, req *msg.PredictRequest) {
	ok := make(chan struct{}) // closed when the job.id and job.cancel are set and it is safe to process the next request.
	select {
	case <-ctx.Done():
		w.reject(ctx, reply, msg.ErrShuttingDown, `worker shutting down`)
	case w.job.ch <- predictRequest{ok, reply, job, req}:
		select {
		case <-ok:
		case <-ctx.Done():
		}
		// do nothing, the model will handle response.
	default:
		w.reject(ctx, reply, msg.ErrBusy, `worker busy`)
	}
}

func (w *worker) interrupt(ctx context.Context, reply, job string) {
	w.job.control.Lock()
	defer w.job.control.Unlock()
	if w.job.id != job {
		return // not our job.
	}
	w.job.cancel()
	w.job.id = ``
	_ = w.respond(ctx, reply, &msg.WorkerResponse{Interrupt: &msg.InterruptResponse{}})
}

func (w *worker) processPredictRequests() {
	for req := range w.job.ch {
		slog.Debug(`processing predict request`, `job`, req.job, `reply`, req.reply)
		w.processPredictRequest(req)
		slog.Debug(`finished processing predict request`, `job`, req.job, `reply`, req.reply)
	}
}

func (w *worker) processPredictRequest(req predictRequest) {
	w.job.control.Lock()
	w.job.id = req.job
	ctx, cancel := context.WithCancel(context.Background())
	w.job.cancel = cancel
	w.job.control.Unlock()
	close(req.ok)

	for _, hook := range w.hooks {
		err := hook(ctx, w.model, req.PredictRequest)
		switch err := err.(type) {
		case nil:
			// do nothing
		case msg.Error:
			w.reject(ctx, req.reply, err.Code, err.Err)
			return
		default:
			w.reject(ctx, req.reply, msg.ErrHookFailed, err.Error())
			return
		}
	}

	callback := func(llm.Prediction) error { return nil }
	reply := req.reply
	if req.Stream != `` {
		callback = func(p llm.Prediction) error {
			output := p.String()
			if output == `` {
				return nil // weird hiccup, we should run this down.
			}
			// TODO: capture errors and send them back through the driver.
			return w.respond(ctx, req.Stream, &msg.WorkerResponse{
				Job:    req.job,
				Stream: &msg.StreamResponse{Output: output},
			})
		}
		// send an empty response to indicate that we will be streaming.
		err := w.respond(ctx, req.reply, &msg.WorkerResponse{Job: req.job})
		if err != nil {
			return // do not continue if we cannot respond.
		}
		reply = req.Stream
	}

	output, err := w.model.Predict(ctx, req.Input, callback)
	switch {
	case err == nil:
		_ = w.respond(ctx, reply, &msg.WorkerResponse{
			Job:     req.job,
			Predict: &msg.PredictResponse{Output: output},
		})
	case errors.Is(err, context.Canceled):
		w.reject(ctx, reply, msg.ErrInterrupted, `prediction interrupted`)
	default:
		w.reject(ctx, reply, msg.ErrPredictionFailed, err.Error())
	}
}

type predictRequest struct {
	ok    chan struct{} // closed when the job.id and job.cancel are set and it is safe to process the next request.
	reply string
	job   string
	*msg.PredictRequest
}

func (w *worker) reject(ctx context.Context, reply string, code int, message string) {
	_ = w.respond(ctx, reply, &msg.WorkerResponse{
		Error: &msg.Error{
			Code: code,
			Err:  message,
		},
	})
}

func (w *worker) respond(ctx context.Context, subject string, resp *msg.WorkerResponse) error {
	data, err := json.Marshal(resp)
	if err != nil {
		panic(err)
	}
	err = w.conn.Publish(subject, data)
	if err != nil {
		slog.From(ctx).Error(`failed to publish response`, `subject`, subject, `err`, err)
	}
	return err
}

// An Option is a function that alters a worker's behavior.
type Option func(*worker)

// Predictor sets the model to use for predictions.
func Predictor(model llm.Predictor) Option {
	return func(w *worker) {
		if w.model != nil {
			w.err = errors.New("only one predictor can be used by a worker")
		}
		w.model = model
	}
}

// A Hook is a function that is called before a prediction is made, allowing it to alter the request.  This is useful
// when a worker can do some preprocessing of input data, such as reducing the input length to fit context.
func Hook(hook func(ctx context.Context, model llm.Predictor, req *msg.PredictRequest) error) Option {
	return func(w *worker) { w.hooks = append(w.hooks, hook) }
}

// Conn sets the NATS connection to use for getting requests and publishing responses.  This is an alternative to
// letting the worker manage its own connection.
func Conn(conn *nats.Conn) Option {
	return func(w *worker) {
		if w.conn != nil {
			w.err = errors.New("only one NATS connection is used by a worker")
		}
		w.conn = conn
	}
}
