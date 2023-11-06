package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/rs/zerolog"
	zlog "github.com/rs/zerolog/log"
	"github.com/rs/zerolog/pkgerrors"
	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/zugzug-go"
	"github.com/swdunlop/zugzug-go/zug/parser"
)

func init() {
	zlog.Logger = zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr}).
		With().Timestamp().Logger()
	zerolog.DefaultContextLogger = &zlog.Logger
	zerolog.ErrorStackMarshaler = pkgerrors.MarshalStack

	log := zerolog.New(zerolog.ConsoleWriter{Out: os.Stdout, TimeFormat: `2006-01-02 15:04:05`}).With().Timestamp().Logger()
	zerolog.DefaultContextLogger = &log
	zlog.Logger = log
}

func main() {
	sort.Slice(tasks, func(i, j int) bool { return tasks[i].Name < tasks[j].Name })
	zugzug.Main(tasks)
}

var (
	cfgModel       string
	cfgModelOption = parser.String(&cfgModel, `model`, `m`, `model to use`)
	tasks          = zugzug.Tasks{
		{Name: `predict`, Fn: runPredict, Use: `runs a prediction`, Parser: parser.New(
			cfgModelOption,
		)},
		{Name: `encode`, Fn: runEncode, Use: `encodes text as tokens`, Parser: parser.New(
			cfgModelOption,
		)},
		{Name: `decode`, Fn: runDecode, Use: `decodes tokens as text`, Parser: parser.New(
			cfgModelOption,
		)},
	}
)

func runEncode(ctx context.Context) error {
	return withModel(func(m llm.Model) error {
		m, err := llm.New(cfgModel)

		tokens := m.Encode(strings.Join(parser.Args(ctx), ` `))
		if len(tokens) == 0 {
			return nil
		}
		_, err = fmt.Fprintf(os.Stdout, `%d`, tokens[0])
		if err != nil {
			return err
		}
		for _, token := range tokens[1:] {
			_, err = fmt.Fprintf(os.Stdout, ` %d`, token)
			if err != nil {
				return err
			}
		}
		_, err = fmt.Fprintln(os.Stdout)
		return err
	})
}

func runDecode(ctx context.Context) error {
	args := parser.Args(ctx)
	tokens := make([]llm.Token, len(args))
	for i, arg := range args {
		v, err := strconv.Atoi(arg)
		if err != nil {
			return err
		}
		tokens[i] = llm.Token(v)
	}
	return withModel(func(m llm.Model) error {
		_, err := os.Stdout.WriteString(m.Decode(tokens))
		return err
	})
}

func runPredict(ctx context.Context) error {
	return withModel(func(m llm.Model) error {
		tokens := m.Encode(strings.Join(parser.Args(ctx), ` `))
		s, err := m.Predict(tokens)
		if err != nil {
			return err
		}
		defer s.Close()
		for {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			t, err := s.Next(nil)
			switch err {
			case nil:
				_, err = os.Stdout.WriteString(m.Decode([]llm.Token{t}))
				if err != nil {
					return err
				}
				_ = os.Stdout.Sync()
			case io.EOF:
				return nil
			case llm.Overload{}:
				return fmt.Errorf(`prediction exceeds model capacity`)
			default:
				return err
			}
		}
	})
}

func withModel(process func(llm.Model) error) error {
	m, err := llm.New(cfgModel)
	if err != nil {
		return err
	}
	defer m.Close()
	process(m)
	return nil
}
