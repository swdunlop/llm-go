package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/chzyer/readline"
	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/llm-go/configuration"
	"github.com/swdunlop/llm-go/internal/slog"
	"github.com/swdunlop/llm-go/nats/worker"
	"github.com/swdunlop/zugzug-go"

	"github.com/swdunlop/llm-go/llama"
	"github.com/swdunlop/llm-go/nats"
)

var tasks = zugzug.Tasks{
	{Name: "worker", Use: "runs a NATS worker with the specified LLM type and model", Fn: runWorker},
	{Name: "client", Use: "runs the specified LLM type and model in interactive mode", Fn: runClient},
	{Name: "predict", Use: "reads stdin and predicts a continuation to stdout", Fn: predictOutput},
	{Name: "settings", Use: "prints the current settings combining the defaults and configuration", Fn: printSettings},
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
	output, err := m.Predict(ctx, strings.Split(input, "\n"), func(prediction llm.Prediction) error {
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
	var driver string
	err := configuration.Get(&driver, cf, `type`)
	if err != nil {
		return nil, err
	}
	return llm.New(driver, cf)
}

func printSettings(ctx context.Context) error {
	cf := configuration.Overlay{
		cf,
		asConfiguration(llama.ModelDefaults(defaultModel)),
		asConfiguration(llama.PredictDefaults()),
		asConfiguration(nats.ClientDefaults()),
	}
	items := cf.Configured()
	sort.Strings(items)
	for _, item := range items {
		values := cf.GetConfiguration(item)
		switch len(values) {
		case 0:
			// do nothing
		case 1:
			fmt.Printf("llm_%s=%q\n", item, values[0])
		default:
			fmt.Printf("llm_%s:\n", item)
			for _, v := range values {
				fmt.Printf("  %q\n", v)
			}
		}
	}
	return nil
}

func asConfiguration(v any) configuration.Interface {
	cf, ok := v.(configuration.Interface)
	if ok {
		return cf
	}
	var err error
	cf, err = configuration.Marshal(v)
	if err != nil {
		panic(err)
	}
	return cf
}

var cf = configuration.Overlay{
	configuration.Environment(`llm_`),
	configuration.Map{
		`type`:  {defaultType},
		`model`: {defaultModel},
	},
}

const (
	defaultType  = `llama`
	defaultModel = `vicuna-7b-v1.5.ggmlv3.q5_K_M.bin`
)
