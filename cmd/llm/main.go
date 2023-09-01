package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/chzyer/readline"
	"github.com/mitchellh/go-wordwrap"
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
	driver, ok := cf[`type`]
	if !ok {
		return nil, fmt.Errorf(`you must specify an llm_type or leave it unset`)
	}
	return llm.New(fmt.Sprint(driver), cf)
}

func printSettings(ctx context.Context) error {
	options := make(map[string]llm.Option, 256)
	names := make([]string, 0, 256)

	options[`type`] = llm.Option{
		Name:  `type`,
		Value: defaultType,
		Use:   `The type of LLM to use, such as "nats" or "llama."`,
	}
	names = append(names, `type`)

	for _, driver := range []string{`nats`, `llama`} {
		driverOptions := llm.Settings(driver)
		for _, opt := range driverOptions {
			if _, dup := options[opt.Name]; !dup {
				names = append(names, opt.Name)
			}
			options[opt.Name] = opt
		}
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	defer tw.Flush()
	for _, name := range names {
		option := options[name]
		value := os.Getenv(`llm_` + name)
		var val, def []byte
		var err error
		if option.Value != nil {
			def, err = json.Marshal(option.Value)
			if err != nil {
				panic(err)
			}
		}
		if value != "" {
			val, err = json.Marshal(value)
			if err != nil {
				panic(err)
			}
		}
		switch {
		case val != nil && def != nil && !bytes.Equal(val, def):
			option.Use += fmt.Sprintf(" (current: %s, default: %s)", val, def)
		case val != nil:
			option.Use += fmt.Sprintf(" (current: %s)", val)
		case def != nil:
			option.Use += fmt.Sprintf(" (default: %s)", def)
		}
		option.Use = strings.ReplaceAll(wordwrap.WrapString(option.Use, 60), "\n", "\n\t")
		if strings.ContainsAny(option.Use, "\n") {
			option.Use += "\n\t"
		}
		fmt.Fprintf(tw, "llm_%s\t%s\n", name, option.Use)
	}
	return nil
}

var cf = envConfig()

func envConfig() map[string]any {
	cf := llm.Env(`llm_`)
	if _, ok := cf[`type`]; !ok {
		cf[`type`] = defaultType
	}
	return cf
}

const (
	defaultType = `llama`
)
