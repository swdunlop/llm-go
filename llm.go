package llm

import (
	"github.com/rs/zerolog"
	"github.com/swdunlop/llm-go/internal/llama"
	"github.com/swdunlop/llm-go/predict"
)

// New loads an LLM model using the provided options from a language model stored at the given path.  A single model
// can be used to generate multiple prediction streams concurrently.
func New(modelPath string, options ...Option) (Model, error) {
	var cfg config
	var err error
	cfg.logger = zerolog.Nop()
	cfg.llama, err = llama.Load(modelPath)
	if err != nil {
		return nil, err
	}
	cfg.parameters = llama.Defaults()
	for _, option := range options {
		option(&cfg)
	}
	if cfg.err != nil {
		cfg.llama.Close()
		return nil, cfg.err
	}
	return &cfg, nil
}

// Zerolog specifies a logger to use for the model and its predictions.  If not specified,  the default logger will be
// used.
func Zerolog(logger zerolog.Logger) Option {
	return func(cfg *config) {
		cfg.logger = logger
	}
}

type Model interface {
	// Close releases any resources held by the language model.  This should not be called while a stream has not been
	// closed.
	Close()

	// Encode converts a string into a sequence of tokens that can be used to predict the next token in the sequence.
	Encode(string) []Token

	// Decode converts a sequence of tokens into a string.
	Decode([]Token) string

	// Predict returns a prediction stream that can be used to generate text.
	Predict(tokens []Token, options ...predict.Option) (Stream, error)

	// Load loads the current state of the language model from a file.
	// Load(path string) error
}

// A Stream is a prediction stream from a language model that can be used to generate text.
type Stream interface {
	Close()

	// Pos returns the current position in the stream.
	// Pos() int

	// Reset resets the prediction stream to a previous position; this will panic if the position is less than zero or
	// greater than the current position.
	// Reset(pos int)

	// Next optionally inputs more tokens into the prediction stream and then returns the next token in the stream.
	// Next will return io.EOF if the model indicated the end of the stream (EOS).
	// Next may indicate Overload if the model is overloaded and cannot accept more tokens and must instead be
	// reset.
	Next(tokens []Token) (Token, error)

	// Save saves the current state of the prediction stream to a file.
	// Save(path string) error
}

// Predict applies a set of prediction options to the model to establish defaults.
func Predict(options ...predict.Option) Option {
	return func(cfg *config) {
		cfg.parameters, cfg.err = predict.Parameters(&cfg.parameters, options...)
	}
}

// An Option affects the configuration of a language model.
type Option func(*config)

type config struct {
	llama      llama.Model
	parameters llama.Parameters
	logger     zerolog.Logger
	err        error
}

func (cfg *config) Close() { cfg.llama.Close() }

func (cfg *config) Encode(text string) []Token   { return cfg.llama.Encode(text) }
func (cfg *config) Decode(tokens []Token) string { return cfg.llama.Decode(tokens) }

func (cfg *config) Predict(tokens []Token, options ...predict.Option) (Stream, error) {
	var stream stream
	parameters, err := predict.Parameters(&cfg.parameters, options...)
	if err != nil {
		return nil, err
	}
	stream.llama, err = cfg.llama.Predict(&cfg.logger, &parameters, tokens)
	if err != nil {
		return nil, err
	}
	return stream, nil
}

type stream struct{ llama llama.Stream }

func (st stream) Close() { st.llama.Close() }
func (st stream) Next(tokens []Token) (Token, error) {
	return st.llama.Next(tokens)
}

// A Token expresses part of the text in a language model.  Tokens may be specific to their model or family of models.
type Token = llama.Token

// Overload is returned by Predict and Next when the number of tokens in the stream exceeds the model's capacity.
type Overload struct{}

// Error implements the Go error interface by simply stating "model overload" as the error message.
func (Overload) Error() string { return `model overload` }
