package main

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

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
	options := llama.DefaultOptions()
	enc.Encode(options)
	llm, err := llama.New(os.Args[1], nil, &options)
	if err != nil {
		return err
	}
	defer func() {
		if llm != nil {
			llm.Close()
		}
	}()
	for {
		var req Request
		req.Options = options
		err := dec.Decode(&req)
		switch err {
		case nil:
			ret := process(ctx, enc, llm, &req)
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

func process(ctx context.Context, enc *json.Encoder, llm llama.Interface, req *Request) (rsp *Response) {
	rsp = new(Response)
	start := time.Now()
	var err error
	defer func() {
		rsp.Seconds = time.Since(start).Seconds()
		if err != nil {
			rsp.Error = err.Error()
		}
	}()
	system := llm.Encode(req.System)
	if req.Options.NumKeep < 0 {
		req.Options.NumKeep = len(system)
	}
	response := make([]string, 0, 256)
	prompt := append(system, llm.Encode(req.Prompt)...)
	if len(prompt) == 0 {
		rsp.Error = "missing prompt and system"
		return
	}
	// TODO(swdunlop): correct truncated prompt to the next newline to avoid orphan tokens.
	llm.Predict(ctx, prompt, func(p *llama.Prediction) {
		if len(p.Response) > 0 {
			enc.Encode(p.Response)
			response = append(response, p.Response)
		}
	})
	rsp.Response = strings.Join(response, "")
	return
}

type interrupted struct{}

type Request struct {
	System  string        `json:"system"`  // the tokens from system are always used.
	Prompt  string        `json:"prompt"`  // the tokens from prompt are used, preferring the later ones.
	Options llama.Options `json:"options"` // options for generation, model options are ignored.
}

type Response struct {
	Response string  `json:"response,omitempty"`
	Error    string  `json:"error,omitempty"`
	Seconds  float64 `json:"seconds"`
}
