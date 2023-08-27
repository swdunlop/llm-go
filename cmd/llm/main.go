package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/chzyer/readline"
	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/llm-go/configuration"
	"github.com/swdunlop/llm-go/internal/slog"
	"github.com/swdunlop/llm-go/nats/worker"
	"github.com/swdunlop/zugzug-go"

	_ "github.com/swdunlop/llm-go/llama"
	_ "github.com/swdunlop/llm-go/nats"
)

var tasks = zugzug.Tasks{
	{Name: "worker", Use: "runs a NATS worker with the specified LLM type and model", Fn: runWorker},
	{Name: "client", Use: "runs the specified LLM type and model in interactive mode", Fn: runClient},
	{Name: "predict", Use: "reads stdin and predicts a continuation to stdout", Fn: predictOutput},
}

func init() {
	slog.Init(os.Stderr)
}

func main() {
	zugzug.Main(tasks)
}

func predictOutput(ctx context.Context) error {
	m, err := newPredictor(ctx)
	if err != nil {
		return err
	}
	defer m.Release()

	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		return err
	}
	return outputPrediction(ctx, os.Stdout, m, string(input))
}

func runWorker(ctx context.Context) error {
	return worker.Run(ctx, cf)
}

func runClient(ctx context.Context) error {
	m, err := newPredictor(ctx)
	if err != nil {
		return err
	}
	defer m.Release()

	rl, err := readline.New(`> `)
	if err != nil {
		return err
	}
	defer rl.Close()
	stdout := rl.Stdout()

	for {
		line, err := rl.Readline()
		if err != nil {
			return err
		}
		err = outputPrediction(ctx, stdout, m, line)
		if err != nil {
			return err
		}
	}
}

func outputPrediction(ctx context.Context, w io.Writer, m llm.Predictor, input string) error {
	predicted := 0
	output, err := m.Predict(ctx, input, func(prediction llm.Prediction) error {
		output := prediction.String()
		predicted += len(output)
		_, err := w.Write([]byte(output))
		os.Stdout.Sync() // TODO
		return err
	})
	// not all LLMs support incremental prediction, so we check to see if there is unprinted output and
	// print it.
	if len(output) > predicted {
		_, _ = w.Write([]byte(output[predicted:]))
	}
	if !strings.HasSuffix(output, "\n") {
		_, err = w.Write([]byte("\n"))
	}
	return err
}

func newPredictor(ctx context.Context) (llm.Predictor, error) {
	m, err := newLLM(ctx)
	if err != nil {
		return nil, err
	}
	predictor, ok := m.(llm.Predictor)
	if !ok {
		m.Release()
		return nil, fmt.Errorf(`%q does not implement llm.Predictor`, m)
	}
	return predictor, nil
}

func newLLM(ctx context.Context) (llm.Interface, error) {
	var model string
	err := configuration.Get(&model, cf, `llm_model`)
	if err != nil {
		return nil, err
	}
	return llm.New(model, cf)
}

var cf = configuration.Overlay{
	configuration.Environment(``),
	configuration.Map{
		`llm_type`: {defaultType},
		`model`:    {defaultModel},
	},
}

const (
	defaultType  = `llama`
	defaultModel = `vicuna-7b-v1.5.ggmlv3.q5_K_M.bin`
)
