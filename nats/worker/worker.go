// Package worker implements a NATS-based worker for large language models.
package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"hash/fnv"
	"os"
	"strings"
	"sync"

	"github.com/nats-io/nats.go"
	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/llm-go/internal/slog"
	client "github.com/swdunlop/llm-go/nats"
	msg "github.com/swdunlop/llm-go/nats/protocol"
)

// Run will run a NATS-based worker with the provided NATS connection and configuration.
func Run(ctx context.Context, cf map[string]any, options ...Option) error {
	slog.From(ctx).Debug(`starting worker`)
	var w worker
	w.cf = cf
	w.options.Options = client.Defaults()
	w.options.ClientName = `llm-worker`
	w.options.Dir = `.`
	err := llm.Unmap(cf, &w.options)
	if err != nil {
		return err
	}
	for _, opt := range options {
		opt(&w)
		if w.err != nil {
			return err
		}
	}

	if w.options.LLMType == "" {
		return errors.New(`llm_type must be specified`)
	}

	w.driverOptions = llm.Settings(w.options.LLMType)
	if w.driverOptions == nil {
		return errors.New(`llm_type not found`)
	}

	if w.conn == nil {
		var err error
		w.conn, err = w.options.Dial() // TODO: error handler
		if err != nil {
			return err
		}
		defer w.conn.Close()
	}

	defer func() {
		if w.model.release != nil {
			w.model.release()
		}
	}()

	ch := make(chan *nats.Msg)
	slog.From(ctx).Debug(`subscribing to worker subject`, `subject`, w.options.WorkerSubject)
	sub, err := w.conn.ChanQueueSubscribe(w.options.WorkerSubject, "llm-worker", ch)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		defer close(ch)
		<-ctx.Done()
		err := sub.Unsubscribe()
		if err != nil {
			slog.Warn(`failed to unsubscribe from worker subject`, `subject`, w.options.WorkerSubject, `err`, err.Error())
		}
	}()

	w.job.done = make(chan struct{})
	w.job.ch = make(chan predictRequest, 4)
	go w.processPredictRequests(ctx)
	w.run(ctx, ch)
	return nil
}

// Options describes the configuration options for a NATS-based worker.  Note that this configuration will be passed on
// to the LLM driver, so the driver may have additional configuration options.
type Options struct {
	client.Options

	// LLM implementation to use.
	LLMType string `json:"type"`

	// Dir provides the directory where models are stored, defaults to "."
	Dir string `json:"dir,omitempty"`
}

type worker struct {
	cf            map[string]any
	options       Options
	driverOptions []llm.Option
	hooks         []func(ctx context.Context, model llm.Interface, req *msg.PredictRequest) error
	conn          *nats.Conn
	job           struct {
		done    chan struct{}
		ch      chan predictRequest
		control sync.RWMutex
		id      string
		cancel  func() // cancels a prediction
	}
	model struct {
		hash    []byte // last configuration used to create the model.
		llm     llm.Interface
		release func()
	}
	err error // used by hooks to indicate a fatal error.
}

func (w *worker) run(ctx context.Context, ch chan *nats.Msg) {
	defer func() { <-w.job.done }()
	defer close(w.job.ch)
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
		w.reject(ctx, nm.Reply, ``, msg.ErrIllegibleRequest, err.Error())
	case req.Job == "":
		w.reject(ctx, nm.Reply, ``, msg.ErrInvalidRequest, "job id is required")
		return
	}

	ok := false
	if req.List != nil {
		w.list(ctx, nm.Reply, req.Job)
		ok = true
	}
	if req.Interrupt != nil {
		w.interrupt(ctx, nm.Reply, req.Job)
		ok = true
	}
	if req.Predict != nil {
		w.predict(ctx, nm.Reply, req.Job, req.Predict)
		ok = true
	}

	if !ok {
		w.reject(ctx, nm.Reply, req.Job, msg.ErrUnsupportedCommand, "command not supported")
	}
}

// predict must be done out of process to avoid blocking the NATS worker.
func (w *worker) predict(ctx context.Context, reply, job string, req *msg.PredictRequest) {
	ok := make(chan struct{}) // closed when the job.id and job.cancel are set and it is safe to process the next request.
	select {
	case w.job.ch <- predictRequest{ok, reply, job, req}:
		select {
		case <-ok:
		case <-ctx.Done():
		}
	case <-ctx.Done():
		w.reject(ctx, reply, job, msg.ErrShuttingDown, `worker shutting down`) // should never happen.
	case <-w.job.done:
		w.reject(ctx, reply, job, msg.ErrShuttingDown, `worker shut down`) // should never happen.
	default:
		w.reject(ctx, reply, job, msg.ErrBusy, `worker busy`)
	}
}

func (w *worker) list(ctx context.Context, reply, job string) {
	options := w.driverOptions
	entries, err := os.ReadDir(w.options.Dir)
	if err != nil {
		w.reject(ctx, reply, job, msg.ErrUnknown, err.Error())
		return
	}
	models := make([]msg.ModelInfo, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if len(name) < 1 || name[0] == '.' || !strings.HasSuffix(name, `.bin`) {
			continue
		}
		// NOTE: we do not currently have a way to get options from a model, all we have are the options from the driver.
		// After GGUF, we may.
		models = append(models, msg.ModelInfo{
			Model:   name,
			Options: options,
		})
	}
	_ = w.respond(ctx, reply, &msg.WorkerResponse{
		Job: job,
		List: &msg.ListResponse{
			Models: models,
		},
	})
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

func (w *worker) processPredictRequests(ctx context.Context) {
	defer close(w.job.done)
	for req := range w.job.ch {
		slog.Debug(`processing predict request`, `job`, req.job, `reply`, req.reply)
		w.processPredictRequest(ctx, req)
		slog.Debug(`finished processing predict request`, `job`, req.job, `reply`, req.reply)
	}
}

func (w *worker) processPredictRequest(ctx context.Context, req predictRequest) {
	w.job.control.Lock()
	w.job.id = req.job
	ctx, cancel := context.WithCancel(ctx) // TODO: this should be a child of the context passed to run
	w.job.cancel = cancel
	w.job.control.Unlock()
	close(req.ok)

	err := w.loadModel(ctx, req.Options)
	if err != nil {
		// TODO: detect if the model is not found and return a different error.
		w.reject(ctx, req.reply, req.job, msg.ErrInvalidModel, err.Error())
		return
	}

	for _, hook := range w.hooks {
		err := hook(ctx, w.model.llm, req.PredictRequest)
		switch err := err.(type) {
		case nil:
			// do nothing
		case msg.Error:
			w.reject(ctx, req.reply, req.job, err.Code, err.Err)
			return
		default:
			w.reject(ctx, req.reply, req.job, msg.ErrHookFailed, err.Error())
			return
		}
	}

	callback := func(llm.Prediction) error {
		return ctx.Err() // keep checking for context cancellation.
	}
	reply := req.reply
	if req.Stream != `` {
		callback = func(p llm.Prediction) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			output := p.String()
			if output == `` {
				return nil // weird hiccup, we should run this down.
			}
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

	output, err := w.model.llm.Predict(ctx, req.Options, req.Input, callback)
	switch {
	case err == nil:
		_ = w.respond(ctx, reply, &msg.WorkerResponse{
			Job:     req.job,
			Predict: &msg.PredictResponse{Output: output},
		})
	case errors.Is(err, context.Canceled):
		w.reject(ctx, reply, req.job, msg.ErrInterrupted, `prediction interrupted`)
	default:
		w.reject(ctx, reply, req.job, msg.ErrPredictionFailed, err.Error())
	}
}

func (w *worker) loadModel(ctx context.Context, cf map[string]any) error {
	h := fnv.New128a()
	enc := json.NewEncoder(h)
	c2 := make(map[string]any, len(w.cf))
	for _, option := range w.driverOptions {
		if !option.Init {
			continue
		}
		value := option.Value
		if str, ok := cf[option.Name]; ok {
			value = str
		} else if str, ok := w.cf[option.Name]; ok {
			value = str
		}
		c2[option.Name] = value
		enc.Encode(option.Name)
		enc.Encode(value)
	}
	hash := h.Sum(nil)
	cf = c2

	if w.model.llm != nil && bytes.Equal(hash, w.model.hash) {
		return nil // already loaded.
	}

	for name, value := range w.cf {
		if _, ok := cf[name]; !ok {
			cf[name] = value
		}
	}

	if model, ok := cf[`model`].(string); ok {
		if strings.IndexAny(model, `/\`) >= 0 {
			return errors.New(`model name cannot contain path separators`)
		}
	}

	m, err := llm.New(w.options.LLMType, cf)
	if err != nil {
		return err
	}
	w.model.llm = m
	w.model.release = m.Release
	w.model.hash = hash
	return nil
}

type predictRequest struct {
	ok    chan struct{} // closed when the job.id and job.cancel are set and it is safe to process the next request.
	reply string
	job   string
	*msg.PredictRequest
}

func (w *worker) reject(ctx context.Context, reply string, job string, code int, message string) {
	_ = w.respond(ctx, reply, &msg.WorkerResponse{
		Job: job,
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

// A Hook is a function that is called before a prediction is made, allowing it to alter the request.  This is useful
// when a worker can do some preprocessing of input data, such as reducing the input length to fit context.
func Hook(hook func(ctx context.Context, model llm.Interface, req *msg.PredictRequest) error) Option {
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
