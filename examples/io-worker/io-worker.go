package main

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"os/signal"
	"time"

	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/llm-go/configuration"
	"github.com/swdunlop/llm-go/llama"
)

func main() {
	err := runWorker()
	if err != nil {
		println(`!!`, err.Error())
		os.Exit(1)
	}
}

func runWorker() error {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	go func() { <-ctx.Done(); os.Stdin.Close() }()
	enc := json.NewEncoder(os.Stdout)
	dec := json.NewDecoder(os.Stdin)
	model, err := llama.New(configuration.Map{
		"model": {os.Args[1]},
	})
	if err != nil {
		return err
	}
	defer model.Release()
	for {
		var req Request
		err := dec.Decode(&req)
		switch err {
		case nil:
			ret := process(ctx, enc, model, &req)
			err := enc.Encode(ret)
			if err != nil {
				return err
			}
		case io.EOF:
			return nil
		default:
			if err := ctx.Err(); err != nil {
				return err // do not nag.
			}
			return err
		}
	}
}

func process(ctx context.Context, enc *json.Encoder, model llm.Predictor, req *Request) (rsp *Response) {
	rsp = new(Response)
	start := time.Now()
	var err error
	defer func() {
		rsp.Seconds = time.Since(start).Seconds()
		if err != nil {
			rsp.Error = err.Error()
		}
	}()
	rsp.Response, err = model.Predict(ctx, req.Prompt, func(p llm.Prediction) error {
		str := p.String()
		if str != "" {
			enc.Encode(str)
		}
		return nil
	})
	return
}

type Request struct {
	System string `json:"system"` // the tokens from system are always used.
	Prompt string `json:"prompt"` // the tokens from prompt are used, preferring the later ones.
	// TODO(swdunlop): add options
}

type Response struct {
	Response string  `json:"response,omitempty"`
	Error    string  `json:"error,omitempty"`
	Seconds  float64 `json:"seconds"`
}
