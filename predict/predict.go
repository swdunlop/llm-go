package predict

import (
	"encoding/json"

	"github.com/swdunlop/llm-go/internal/llama"
)

// Parameters returns the parameters for a prediction stream that obeys the given options.
func Parameters(base *llama.Parameters, options ...Option) (llama.Parameters, error) {
	parameters := *base
	cfg := config{parameters: &parameters}
	for _, option := range options {
		option(&cfg)
		if cfg.err != nil {
			return parameters, cfg.err
		}
	}
	return parameters, nil
}

// Seed specifies the seed for a prediction stream.
func Seed(seed int) Option {
	return func(cfg *config) {
		cfg.parameters.Seed = uint32(seed)
	}
}

// Temperature specifies the sampling temperature for a prediction stream.
func Temperature(temperature float64) Option {
	return func(cfg *config) {
		cfg.parameters.Temperature = float32(temperature)
	}
}

// JSON applies the JSON used by examples/server in llama.cpp as options to a prediction stream.
func JSON(msg []byte) Option {
	return func(cfg *config) {
		cfg.err = json.Unmarshal(msg, cfg.parameters)
	}
}

// Environ applies environment variables with the provided prefix as options to a prediction stream.
// func Environ(prefix string) Option {
// 	panic(`TODO`)
// }

// An Option is an option for a prediction stream.
type Option func(*config)

type config struct {
	parameters *llama.Parameters
	err        error
}
