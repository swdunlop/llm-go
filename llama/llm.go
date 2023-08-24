package llama

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/pbnjay/memory"
)

type Interface interface {
	Predict(context.Context, []int, func(*Prediction)) error
	Embedding(string) ([]float64, error)
	Encode(string) []int
	Decode(...int) string
	SetOptions(Options) // TODO(swdunlop): move Options to a map[string]any
	Close()
}

func New(model string, adapters []string, opts *Options) (Interface, error) {
	// TODO(swdunlop): move Options to a map[string]any
	// TODO(swdunlop): move adapters ito Options
	// TODO(swdunlop): split prediction and model options apart.
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	//TODO(swdunlop): do we need decodeGGML at all? shouldn't we just use the llama.cpp API for this?
	//TODO(swdunlop): if we do, we should move this to a separate package and add gguf as well.
	//TODO(swdunlop): if we don't, how do we load older GGML models? do we have to offer a converter?
	ggml, err := decodeGGML(f, modelFamilyLlama)
	if err != nil {
		return nil, err
	}

	switch ggml.FileType().String() {
	case "F32", "F16", "Q5_0", "Q5_1", "Q8_0":
		if opts.NumGPU != 0 {
			// F32, F16, Q5_0, Q5_1, and Q8_0 do not support Metal API and will
			// cause the runner to segmentation fault so disable GPU
			log.Printf("WARNING: GPU disabled for F32, F16, Q5_0, Q5_1, and Q8_0")
			opts.NumGPU = 0
		}
	}

	totalResidentMemory := memory.TotalMemory()
	switch ggml.ModelType() {
	case modelType3B, modelType7B:
		if totalResidentMemory < 8*1024*1024 {
			return nil, fmt.Errorf("model requires at least 8GB of memory")
		}
	case modelType13B:
		if totalResidentMemory < 16*1024*1024 {
			return nil, fmt.Errorf("model requires at least 16GB of memory")
		}
	case modelType30B:
		if totalResidentMemory < 32*1024*1024 {
			return nil, fmt.Errorf("model requires at least 32GB of memory")
		}
	case modelType65B:
		if totalResidentMemory < 64*1024*1024 {
			return nil, fmt.Errorf("model requires at least 64GB of memory")
		}
	}

	switch ggml.ModelFamily() {
	case modelFamilyLlama:
		return newLlama(model, adapters, opts)
	default:
		return nil, fmt.Errorf("unknown ggml type: %s", ggml.ModelFamily())
	}
}

type Prediction struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Response  string    `json:"response,omitempty"`

	Done    bool  `json:"done"`
	Context []int `json:"context,omitempty"`

	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	SampleCount        int           `json:"sample_count,omitempty"`
	SampleDuration     time.Duration `json:"sample_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

func (r *Prediction) Summary() {
	if r.TotalDuration > 0 {
		fmt.Fprintf(os.Stderr, "total duration:       %v\n", r.TotalDuration)
	}

	if r.LoadDuration > 0 {
		fmt.Fprintf(os.Stderr, "load duration:        %v\n", r.LoadDuration)
	}

	if r.SampleCount > 0 {
		fmt.Fprintf(os.Stderr, "sample count:         %d token(s)\n", r.SampleCount)
	}

	if r.SampleDuration > 0 {
		fmt.Fprintf(os.Stderr, "sample duration:      %s\n", r.SampleDuration)
		fmt.Fprintf(os.Stderr, "sample rate:          %.2f tokens/s\n", float64(r.SampleCount)/r.SampleDuration.Seconds())
	}

	if r.PromptEvalCount > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval count:    %d token(s)\n", r.PromptEvalCount)
	}

	if r.PromptEvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval duration: %s\n", r.PromptEvalDuration)
		fmt.Fprintf(os.Stderr, "prompt eval rate:     %.2f tokens/s\n", float64(r.PromptEvalCount)/r.PromptEvalDuration.Seconds())
	}

	if r.EvalCount > 0 {
		fmt.Fprintf(os.Stderr, "eval count:           %d token(s)\n", r.EvalCount)
	}

	if r.EvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "eval duration:        %s\n", r.EvalDuration)
		fmt.Fprintf(os.Stderr, "eval rate:            %.2f tokens/s\n", float64(r.EvalCount)/r.EvalDuration.Seconds())
	}
}
