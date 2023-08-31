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
	"github.com/swdunlop/llm-go/internal/slog"
	_ "github.com/swdunlop/llm-go/llama"
	_ "github.com/swdunlop/llm-go/nats"
	"github.com/swdunlop/llm-go/nats/worker"
	"github.com/swdunlop/zugzug-go"
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
	m, err := newLLM(ctx)
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
	m, err := newLLM(ctx)
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

func outputPrediction(ctx context.Context, w io.Writer, m llm.Interface, input string) error {
	predicted := 0
	output, err := m.Predict(ctx, cf, strings.Split(input, "\n"), func(prediction llm.Prediction) error {
		output := prediction.String()
		predicted += len(output)
		_, err := w.Write([]byte(output))
		os.Stdout.Sync() // TODO
		return err
	})
	if err != nil {
		return err
	}
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

func newLLM(ctx context.Context) (llm.Interface, error) {
	var driver string
	driver, ok := cf[`type`]
	if !ok {
		return nil, fmt.Errorf(`you must specify an llm_type or leave it unset`)
	}
	return llm.New(driver, cf)
}

func printSettings(ctx context.Context) error {
	items := make([]string, 0, len(cf))
	for item := range cf {
		items = append(items, item)
	}
	sort.Strings(items)
	for _, item := range items {
		values := cf[item]
		fmt.Printf("llm_%s=%v\n", item, values)
	}
	return nil
}

var cf = envConfig()

func envConfig() map[string]string {
	cfg := map[string]string{
		`type`:  defaultType,
		`model`: defaultModel,
	}
	for _, env := range os.Environ() {
		if !strings.HasPrefix(env, `llm_`) {
			continue
		}
		env = strings.TrimPrefix(env, `llm_`)
		parts := strings.SplitN(env, `=`, 2)
		if len(parts) != 2 {
			continue
		}
		cfg[parts[0]] = parts[1]
	}
	return cfg
}

const (
	defaultType  = `llama`
	defaultModel = `vicuna-7b-v1.5.ggmlv3.q5_K_M.bin`
)
