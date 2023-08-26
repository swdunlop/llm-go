// Package llama provides a low level interface to llama1 and llama2-like models using llama.cpp.  It is more versatile
// and less user friendly than the higher level llm package interface.
package llama

/*
#cgo CFLAGS: -Ofast -std=c11 -fPIC
#cgo CPPFLAGS: -Ofast -Wall -Wextra -Wno-unused-function -Wno-unused-variable -DNDEBUG -DGGML_USE_K_QUANTS
#cgo CXXFLAGS: -std=c++11 -fPIC
#cgo darwin CPPFLAGS:  -DGGML_USE_ACCELERATE
#cgo darwin,arm64 CPPFLAGS: -DGGML_USE_METAL -DGGML_METAL_NDEBUG
#cgo darwin LDFLAGS: -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
#include <stdlib.h>
#include "llama.h"

struct llama_sample_options
{
	float repeat_penalty;
	float frequency_penalty;
	float presence_penalty;
	float temperature;
	int32_t top_k;
	float top_p;
	float tfs_z;
	float typical_p;
	int mirostat;
	float mirostat_tau;
	float mirostat_eta;
	bool penalize_newline;
};

llama_token llama_sample(
		struct llama_context *ctx,
		struct llama_token_data *candidates,
		size_t n_candidates,
		const llama_token *last_tokens,
		size_t n_last_tokens,
		struct llama_sample_options *opts)
{
	llama_token_data_array candidates_p = {
		candidates,
		n_candidates,
		false,
	};

	struct llama_token_data newline = candidates_p.data[llama_token_nl()];

	llama_sample_repetition_penalty(
		ctx, &candidates_p,
		last_tokens, n_last_tokens,
		opts->repeat_penalty);

	llama_sample_frequency_and_presence_penalties(
		ctx, &candidates_p,
		last_tokens, n_last_tokens,
		opts->frequency_penalty, opts->presence_penalty);

	if (!opts->penalize_newline) {
		candidates_p.data[llama_token_nl()] = newline;
	}

	if (opts->temperature <= 0) {
		return llama_sample_token_greedy(ctx, &candidates_p);
	}

	if (opts->mirostat == 1) {
		int mirostat_m = 100;
		float mirostat_mu = 2.0f * opts->mirostat_tau;
		llama_sample_temperature(ctx, &candidates_p, opts->temperature);
		return llama_sample_token_mirostat(
			ctx, &candidates_p,
			opts->mirostat_tau, opts->mirostat_eta,
			mirostat_m, &mirostat_mu);
	} else if (opts->mirostat == 2) {
		float mirostat_mu = 2.0f * opts->mirostat_tau;
		llama_sample_temperature(ctx, &candidates_p, opts->temperature);
		return llama_sample_token_mirostat_v2(
			ctx, &candidates_p,
			opts->mirostat_tau, opts->mirostat_eta,
			&mirostat_mu);
	} else {
		llama_sample_top_k(ctx, &candidates_p, opts->top_k, 1);
		llama_sample_tail_free(ctx, &candidates_p, opts->tfs_z, 1);
		llama_sample_typical(ctx, &candidates_p, opts->typical_p, 1);
		llama_sample_top_p(ctx, &candidates_p, opts->top_p, 1);
		llama_sample_temperature(ctx, &candidates_p, opts->temperature);
		return llama_sample_token(ctx, &candidates_p);
	}
}
*/
import "C"

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
	"unsafe"

	"github.com/pbnjay/memory"
	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/llm-go/configuration"
)

// Init configure things in llama.cpp that are not specific to the llama interface, specifically `LLAMA_USE_NUMA`.  If
// you want to override runtime defaults, you should call this before calling other functions in this package.
func Init(cf configuration.Interface) error {
	initOnce.Do(func() {
		if cf == nil {
			cf = configuration.Environment(`LLAMA_`)
		}
		cf = configuration.Overlay{cf, configuration.Map{
			`use_numa`:     {`true`},
			`n_gpu_layers`: {`1`}, // 0 is CPU only, 1 enables GPU use on Metal
		}}
		var useNuma bool
		err := configuration.Get(&useNuma, cf, `use_numa`)
		if err != nil {
			initError = err
			return
		}
		C.llama_backend_init(C.bool(useNuma))
		var numGPU int
		err = configuration.Get(&numGPU, cf, `n_gpu_layers`)
		if err != nil {
			initError = err
			return
		}

		defaultParams = C.llama_context_default_params()
		defaultParams.n_gpu_layers = C.int(numGPU)
		ncpu = numCPU()
	})
	return initError
}

var initOnce sync.Once
var initError error
var ncpu = 0
var defaultParams C.struct_llama_context_params

//go:embed ggml-metal.metal
var fs embed.FS

func init() {
	llm.RegisterPredictor(`llama`, func(cf configuration.Interface) (llm.Predictor, error) {
		modelOptions := ModelDefaults(``)
		err := configuration.Unmarshal(modelOptions, cf)
		if err != nil {
			return nil, err
		}
		m, err := New(modelOptions)
		if err != nil {
			return nil, err
		}
		err = configuration.Unmarshal(&m.predictOptions, cf)
		if err != nil {
			return nil, err
		}
		return m, nil
	})
}

const modelFamilyLlama modelFamily = "llama"

type llamaModel struct {
	hyperparameters llamaHyperparameters
}

func (llm *llamaModel) ModelFamily() modelFamily {
	return modelFamilyLlama
}

func (llm *llamaModel) ModelType() modelType {
	switch llm.hyperparameters.NumLayer {
	case 26:
		return modelType3B
	case 32:
		return modelType7B
	case 40:
		return modelType13B
	case 60:
		return modelType30B
	case 80:
		return modelType65B
	}

	// TODO(swdunlop): better to indicate an error than guess.
	// TODO: find a better default
	return modelType7B
}

func (llm *llamaModel) FileType() fileType {
	return llm.hyperparameters.FileType
}

type llamaHyperparameters struct {
	// NumVocab is the size of the model's vocabulary.
	NumVocab uint32

	// NumEmbd is the size of the model's embedding layer.
	NumEmbd uint32
	NumMult uint32
	NumHead uint32

	// NumLayer is the number of layers in the model.
	NumLayer uint32
	NumRot   uint32

	// FileType describes the quantization level of the model, e.g. Q4_0, Q5_K, etc.
	FileType llamaFileType
}

type llamaFileType uint32

const (
	llamaFileTypeF32 llamaFileType = iota
	llamaFileTypeF16
	llamaFileTypeQ4_0
	llamaFileTypeQ4_1
	llamaFileTypeQ4_1_F16
	llamaFileTypeQ8_0 llamaFileType = iota + 2
	llamaFileTypeQ5_0
	llamaFileTypeQ5_1
	llamaFileTypeQ2_K
	llamaFileTypeQ3_K_S
	llamaFileTypeQ3_K_M
	llamaFileTypeQ3_K_L
	llamaFileTypeQ4_K_S
	llamaFileTypeQ4_K_M
	llamaFileTypeQ5_K_S
	llamaFileTypeQ5_K_M
	llamaFileTypeQ6_K
)

func (ft llamaFileType) String() string {
	switch ft {
	case llamaFileTypeF32:
		return "F32"
	case llamaFileTypeF16:
		return "F16"
	case llamaFileTypeQ4_0:
		return "Q4_0"
	case llamaFileTypeQ4_1:
		return "Q4_1"
	case llamaFileTypeQ4_1_F16:
		return "Q4_1_F16"
	case llamaFileTypeQ8_0:
		return "Q8_0"
	case llamaFileTypeQ5_0:
		return "Q5_0"
	case llamaFileTypeQ5_1:
		return "Q5_1"
	case llamaFileTypeQ2_K:
		return "Q2_K"
	case llamaFileTypeQ3_K_S:
		return "Q3_K_S"
	case llamaFileTypeQ3_K_M:
		return "Q3_K_M"
	case llamaFileTypeQ3_K_L:
		return "Q3_K_L"
	case llamaFileTypeQ4_K_S:
		return "Q4_K_S"
	case llamaFileTypeQ4_K_M:
		return "Q4_K_M"
	case llamaFileTypeQ5_K_S:
		return "Q5_K_S"
	case llamaFileTypeQ5_K_M:
		return "Q5_K_M"
	case llamaFileTypeQ6_K:
		return "Q6_K"
	default:
		return "Unknown"
	}
}

// Model combines a llama model with its context.  Models are not safe for concurrent use, particularly while performing
// a prediction.
type Model struct {
	params *C.struct_llama_context_params
	model  *C.struct_llama_model
	tokens *C.struct_llama_context

	last   []C.llama_token
	embd   []C.llama_token
	cursor int

	mu sync.Mutex
	gc bool

	ModelOptions
	predictOptions PredictOptions // only used for Predict, not PredictLlama
}

// ModelOptions controls how the model (and its context) are initialized.  These settings, their names and defaults
// come from llama.cpp -- see https://github.com/ggerganov/llama.cpp/tree/master/examples/main for explanations of
// common settings.
type ModelOptions struct {
	Model    string   `json:"model"`
	Adapters []string `json:"adapters"`

	NumCtx             int     `json:"n_ctx,omitempty"`
	NumBatch           int     `json:"n_batch,omitempty"`
	NumGQA             int     `json:"n_gqa,omitempty"`
	RMSNormEps         float32 `json:"rms_norm_eps,omitempty"`
	NumGPU             int     `json:"n_gpu,omitempty"`
	MainGPU            int     `json:"main_gpu,omitempty"`
	LowVRAM            bool    `json:"low_vram,omitempty"`
	F16KV              bool    `json:"f16_kv,omitempty"`
	LogitsAll          bool    `json:"logits_all,omitempty"`
	VocabOnly          bool    `json:"vocab_only,omitempty"`
	UseMMap            bool    `json:"use_mmap,omitempty"`
	UseMLock           bool    `json:"use_mlock,omitempty"`
	EmbeddingOnly      bool    `json:"embedding_only,omitempty"`
	RopeFrequencyBase  float32 `json:"rope_freq_base,omitempty"`
	RopeFrequencyScale float32 `json:"rope_freq_scale,omitempty"`
	NumThread          int     `json:"n_threads,omitempty"`
}

// PredictOptions controls how the model generates text.  These settings mostly handle how the model's output is
// processed when predicting text.  Like ModelOptions, these names and defaults come from llama.cpp.
type PredictOptions struct {
	Seed             int      `json:"seed,omitempty"`
	NumKeep          int      `json:"keep,omitempty"`
	RepeatLastN      int      `json:"repeat_last_n,omitempty"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
	Temperature      float32  `json:"temperature,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	TFSZ             float32  `json:"tfs_z,omitempty"`
	TypicalP         float32  `json:"typical_p,omitempty"`
	Mirostat         int      `json:"mirostat,omitempty"`
	MirostatTau      float32  `json:"mirostat_tau,omitempty"`
	MirostatEta      float32  `json:"mirostat_eta,omitempty"`
	PenalizeNewline  bool     `json:"penalize_newline,omitempty"`
	Stop             []string `json:"stop,omitempty"`
}

// RuntimeDefault provides a default configuration for llama models and predictions, looking at the runtime
// configuration of the host.  (Specifically, what is its OS, CPU architecture, and number of CPUs.)
func RuntimeDefault() configuration.Interface {
	return configuration.Overlay{
		configuration.Map{
			`n_threads`: []string{strconv.Itoa(numCPU())},
		},
		Defaults(),
	}
}

// Defaults returns a default configuration for llama models and prediction that is not dependent on the host.
func Defaults() configuration.Interface {
	// f := func(v any) []string { return []string{fmt.Sprint(v)} }
	generateDefaultsOnce.Do(func() {
		model := ModelDefaults(``)
		var err error
		generatedDefaults, err = configuration.Marshal(model)
		if err != nil {
			panic(err)
		}
	})
	return generatedDefaults
}

var (
	generateDefaultsOnce sync.Once
	generatedDefaults    configuration.Interface
)

// ModelDefaults returns the llama.cpp defaults parameters for a model, and sets the "Model" field to the provided
// string.
func ModelDefaults(model string) *ModelOptions {
	err := Init(nil)
	if err != nil {
		panic(err)
	}
	return &ModelOptions{
		Model:      model,
		NumCtx:     int(defaultParams.n_ctx),
		NumBatch:   int(defaultParams.n_batch),
		NumGPU:     int(defaultParams.n_gpu_layers),
		NumGQA:     int(defaultParams.n_gqa),
		RMSNormEps: float32(defaultParams.rms_norm_eps),
		// TODO: add tensor_split
		LowVRAM:            bool(defaultParams.low_vram),
		F16KV:              bool(defaultParams.f16_kv),
		LogitsAll:          bool(defaultParams.logits_all),
		VocabOnly:          bool(defaultParams.vocab_only),
		RopeFrequencyBase:  float32(defaultParams.rope_freq_base),
		RopeFrequencyScale: float32(defaultParams.rope_freq_scale),
		UseMMap:            bool(defaultParams.use_mmap),
		UseMLock:           bool(defaultParams.use_mlock),
		EmbeddingOnly:      true, // TODO: bool(defaultParams.embedding),
	}
}

// PredictDefaults returns the default prediction options for llama models.  Note that this has a lot of impact on the
// balance of stability against creativity in the model's output.
func PredictDefaults() *PredictOptions {
	err := Init(nil)
	if err != nil {
		panic(err)
	}
	return &PredictOptions{
		Seed:             -1,
		NumKeep:          -1,
		RepeatLastN:      64,
		RepeatPenalty:    1.1,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.9,
		TFSZ:             1.0,
		TypicalP:         1.0,
		Mirostat:         0,
		MirostatTau:      5.0,
		MirostatEta:      0.1,
		PenalizeNewline:  true,
	}
}

// New creates a new interface from a configuration.
func New(options *ModelOptions) (*Model, error) {
	err := Init(nil)
	if err != nil {
		return nil, err
	}
	if options == nil {
		return nil, errors.New(`model options must be provided`)
	}
	if options.NumThread == 0 {
		options.NumThread = numCPU() // we are outsmarting llama.cpp here, at our peril.
	}
	if options.NumGQA == 0 {
		return nil, errors.New(`n_gqa must be set`)
	}
	llm := new(Model)
	llm.ModelOptions = *options
	llm.predictOptions = *PredictDefaults()

	// TODO(swdunlop): split prediction and model options apart.
	if _, err := os.Stat(llm.ModelOptions.Model); err != nil {
		return nil, err
	}

	f, err := os.Open(llm.ModelOptions.Model)
	if err != nil {
		return nil, err
	}
	defer f.Close() //TODO(swdunlop): we really don't need the file open the whole time we load this.

	//TODO(swdunlop): do we need decodeGGML at all? shouldn't we just use the llama.cpp API for this?
	//TODO(swdunlop): if we do, we should move this to a separate package and add gguf as well.
	//TODO(swdunlop): if we don't, how do we load older GGML models? do we have to offer a converter?
	ggml, err := decodeGGML(f, modelFamilyLlama)
	if err != nil {
		return nil, err
	}

	switch ggml.FileType().String() {
	case "F32", "F16", "Q5_0", "Q5_1", "Q8_0":
		if llm.ModelOptions.NumGPU != 0 {
			// F32, F16, Q5_0, Q5_1, and Q8_0 do not support Metal API and will cause the runner to segmentation fault
			return nil, fmt.Errorf("GPU acceleration not available for GGML model types F32, F16, Q5_0, Q5_1, and Q8_0")
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
	case modelType34B:
		if totalResidentMemory < 38*1024*1024 {
			return nil, fmt.Errorf("model requires at least 38GB of memory")
		}
	case modelType65B:
		if totalResidentMemory < 64*1024*1024 {
			return nil, fmt.Errorf("model requires at least 64GB of memory")
		}
	}

	switch ggml.ModelFamily() {
	case modelFamilyLlama:
	default:
		return nil, fmt.Errorf("unknown ggml type: %s", ggml.ModelFamily())
	}

	params := C.llama_context_default_params()
	params.n_ctx = C.int(llm.NumCtx)
	params.n_batch = C.int(llm.NumBatch)
	params.n_gqa = C.int(llm.NumGQA)
	params.rms_norm_eps = C.float(llm.RMSNormEps)
	params.n_gpu_layers = C.int(llm.NumGPU)
	params.main_gpu = C.int(llm.MainGPU)
	params.low_vram = C.bool(llm.LowVRAM)
	params.f16_kv = C.bool(llm.F16KV)
	params.logits_all = C.bool(llm.LogitsAll)
	params.vocab_only = C.bool(llm.VocabOnly)
	params.use_mmap = C.bool(llm.UseMMap)
	params.use_mlock = C.bool(llm.UseMLock)
	params.embedding = C.bool(llm.EmbeddingOnly)
	params.rope_freq_base = C.float(llm.RopeFrequencyBase)
	params.rope_freq_scale = C.float(llm.RopeFrequencyScale)

	if len(llm.ModelOptions.Adapters) > 0 && llm.UseMMap {
		return nil, fmt.Errorf(`you cannot combine mmap with lora adapters`)
	}

	llm.params = &params

	cModel := C.CString(llm.ModelOptions.Model)
	defer C.free(unsafe.Pointer(cModel))

	llm.model = C.llama_load_model_from_file(cModel, params)
	if llm.model == nil {
		return nil, errors.New("failed to load model")
	}

	llm.tokens = C.llama_new_context_with_model(llm.model, params)
	if llm.tokens == nil {
		return nil, errors.New("failed to create context")
	}

	for _, adapter := range llm.ModelOptions.Adapters {
		cAdapter := C.CString(adapter)
		defer C.free(unsafe.Pointer(cAdapter))

		if retval := C.llama_model_apply_lora_from_file(llm.model, cAdapter, nil, C.int(llm.NumThread)); retval != 0 {
			return nil, fmt.Errorf("failed to load adapter %s", adapter)
		}
	}

	// warm up the model
	bos := []C.llama_token{C.llama_token_bos()}
	C.llama_eval(llm.tokens, unsafe.SliceData(bos), C.int(len(bos)), 0, C.int(llm.ModelOptions.NumThread))
	C.llama_reset_timings(llm.tokens)

	return llm, nil
}

func (m *Model) Release() {
	m.gc = true

	m.mu.Lock()
	defer m.mu.Unlock()

	defer C.llama_free_model(m.model)
	defer C.llama_free(m.tokens)
}

var errNeedMoreData = errors.New("need more data")

// Predict implements the llm.Predictor interface by exchanging strings with tokens and using PredictOptions from
// model instantation time.
func (m *Model) Predict(ctx context.Context, content string, fn func(llm.Prediction) error) (string, error) {
	tokens := make([]int, 0, 4096)
	err := m.PredictLlama(ctx, &m.predictOptions, m.Encode(content), func(p *Prediction) error {
		tokens = append(tokens, p.Response...)
		err := fn(p)
		return err
	})
	return m.Decode(tokens...), err
}

// PredictLlama is a more low level implementation of Predict that lets you specify the prediction options for each
// call.
func (m *Model) PredictLlama(ctx context.Context, options *PredictOptions, tokens []int, fn func(*Prediction) error) error {
	C.llama_reset_timings(m.tokens)
	m.marshalPrompt(options, tokens)
	C.llama_set_rng_seed(m.tokens, C.uint(options.Seed))
	json.NewEncoder(os.Stderr).Encode(options) // TODO

	var p Prediction
	p.model = m
	p.Response = make([]int, 0, 16)
	var b bytes.Buffer
	for {
		token, err := m.next(ctx, options)
		if m.gc {
			return nil
		} else if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return err
		}
		p.Response = append(p.Response, int(token))

		b.WriteString(m.Decode(int(token))) //TODO: this looks suspiciously inefficient.

		if err := m.checkStopConditions(options, b); err != nil {
			if errors.Is(err, io.EOF) {
				break
			} else if errors.Is(err, errNeedMoreData) {
				continue
			}

			return err
		}

		if utf8.Valid(b.Bytes()) || b.Len() >= utf8.UTFMax {
			err := fn(&p)
			if err != nil {
				return err
			}
			b.Reset()
			p.Response = p.Response[0:0:cap(p.Response)]
		}
	}
	return fn(&Prediction{Done: true})
}

func (llm *Model) checkStopConditions(options *PredictOptions, b bytes.Buffer) error {
	for _, stopCondition := range options.Stop {
		if stopCondition == strings.TrimSpace(b.String()) {
			return io.EOF
		} else if strings.HasPrefix(stopCondition, strings.TrimSpace(b.String())) {
			return errNeedMoreData
		}
	}

	return nil
}

func (llm *Model) marshalPrompt(options *PredictOptions, ctx []int) []C.llama_token {
	if options.NumKeep < 0 {
		options.NumKeep = len(ctx)
	}

	cTokens := make([]C.llama_token, len(ctx))
	for i := range ctx {
		cTokens[i] = C.llama_token(ctx[i])
	}

	// min(llm.NumCtx - 4, options.NumKeep)
	if llm.NumCtx-4 < options.NumKeep {
		options.NumKeep = llm.NumCtx - 4
	}

	if len(ctx) >= llm.NumCtx {
		// truncate input
		numLeft := (llm.NumCtx - options.NumKeep) / 2
		truncated := cTokens[:options.NumKeep]
		erasedBlocks := (len(cTokens) - options.NumKeep - numLeft - 1) / numLeft
		truncated = append(truncated, cTokens[options.NumKeep+erasedBlocks*numLeft:]...)
		copy(llm.last, cTokens[len(cTokens)-llm.NumCtx:])

		cTokens = truncated
	} else {
		llm.last = make([]C.llama_token, llm.NumCtx-len(cTokens))
		llm.last = append(llm.last, cTokens...)
	}

	var i int
	for i = 0; i < len(llm.embd) && i < len(cTokens) && llm.embd[i] == cTokens[i]; i++ {
		// noop
	}

	llm.embd = cTokens
	if i == len(cTokens) {
		// evaluate at least one token to generate logits
		i--
	}

	llm.cursor = i
	return cTokens
}

// Encode returns the encoded tokens for the given text, which is model specific.  These tokens can be passed to
// PredictLLama, decoded with Decode, or used to extract embeddings.
func (llm *Model) Encode(text string) []int {
	cPrompt := C.CString(text)
	defer C.free(unsafe.Pointer(cPrompt))

	cTokens := make([]C.llama_token, len(text)+1)
	if n := C.llama_tokenize(llm.tokens, cPrompt, unsafe.SliceData(cTokens), C.int(len(cTokens)), true); n > 0 {
		tokens := make([]int, n)
		for i := range cTokens[:n] {
			tokens[i] = int(cTokens[i])
		}

		return tokens
	}

	return nil
}

// Decode converts a slice of tokens into a string.  Note that this string may not be valid UTF-8.
func (llm *Model) Decode(tokens ...int) string {
	var sb strings.Builder
	for _, token := range tokens {
		sb.WriteString(C.GoString(C.llama_token_to_str(llm.tokens, C.llama_token(token))))
	}
	return sb.String()
}

func (llm *Model) next(ctx context.Context, options *PredictOptions) (C.llama_token, error) {
	llm.mu.Lock()
	defer llm.mu.Unlock()

	if len(llm.embd) >= llm.NumCtx {
		numLeft := (llm.NumCtx - options.NumKeep) / 2
		truncated := llm.embd[:options.NumKeep]
		truncated = append(truncated, llm.embd[len(llm.embd)-numLeft:]...)

		llm.embd = truncated
		llm.cursor = options.NumKeep
	}

	for {
		if llm.gc {
			return 0, io.EOF
		}

		if llm.cursor >= len(llm.embd) {
			break
		}

		numEval := len(llm.embd) - llm.cursor
		if numEval > llm.NumBatch {
			numEval = llm.NumBatch
		}

		if err := ctx.Err(); err != nil {
			return 0, err
		}
		if retval := C.llama_eval(llm.tokens, unsafe.SliceData(llm.embd[llm.cursor:]), C.int(numEval), C.int(llm.cursor), C.int(llm.NumThread)); retval != 0 {
			return 0, fmt.Errorf("llama_eval: %d", retval)
		}

		llm.cursor += numEval
	}

	// TODO: split SampleOptions from PredictOptions
	var sampleOpts C.struct_llama_sample_options
	sampleOpts.repeat_penalty = C.float(options.RepeatPenalty)
	sampleOpts.frequency_penalty = C.float(options.FrequencyPenalty)
	sampleOpts.presence_penalty = C.float(options.PresencePenalty)
	sampleOpts.temperature = C.float(options.Temperature)
	sampleOpts.top_k = C.int(options.TopK)
	sampleOpts.top_p = C.float(options.TopP)
	sampleOpts.tfs_z = C.float(options.TFSZ)
	sampleOpts.typical_p = C.float(options.TypicalP)
	sampleOpts.mirostat = C.int(options.Mirostat)
	sampleOpts.mirostat_tau = C.float(options.MirostatTau)
	sampleOpts.mirostat_eta = C.float(options.MirostatEta)
	sampleOpts.penalize_newline = C.bool(options.PenalizeNewline)

	numVocab := C.llama_n_vocab(llm.tokens)
	logits := unsafe.Slice(C.llama_get_logits(llm.tokens), numVocab)

	// TODO: logit bias

	candidates := make([]C.llama_token_data, numVocab)
	for i := range logits {
		candidates[i] = C.llama_token_data{
			id:    C.int(i),
			logit: logits[i],
			p:     0,
		}
	}

	repeatLastN := options.RepeatLastN
	if len(llm.last) < repeatLastN {
		repeatLastN = len(llm.last)
	}

	if llm.NumCtx < repeatLastN {
		repeatLastN = llm.NumCtx
	}

	lastN := llm.last[len(llm.last)-repeatLastN:]

	token := C.llama_sample(
		llm.tokens,
		unsafe.SliceData(candidates), C.size_t(len(candidates)),
		unsafe.SliceData(lastN), C.size_t(len(lastN)),
		&sampleOpts,
	)

	llm.last = append(llm.last, token)
	llm.embd = append(llm.embd, token)

	if token == C.llama_token_eos() {
		return 0, io.EOF
	}

	return token, nil
}

// Embedding returns the embeddings for the given sequence of encoded tokens.
func (llm *Model) Embedding(tokens []int) ([]float64, error) {
	if !llm.EmbeddingOnly {
		return nil, errors.New("llama: embedding not enabled")
	}

	cTokens := make([]C.llama_token, len(tokens))
	for i := range tokens {
		cTokens[i] = C.llama_token(tokens[i])
	}

	retval := C.llama_eval(llm.tokens, unsafe.SliceData(cTokens), C.int(len(tokens)), 0, C.int(llm.NumThread))
	if retval != 0 {
		return nil, errors.New("llama: eval")
	}

	n := C.llama_n_embd(llm.tokens)
	if n <= 0 {
		return nil, errors.New("llama: no embeddings generated")
	}
	cEmbeddings := unsafe.Slice(C.llama_get_embeddings(llm.tokens), n)

	embeddings := make([]float64, len(cEmbeddings))
	for i, v := range cEmbeddings {
		embeddings[i] = float64(v)
	}
	return embeddings, nil
}

// Prediction contains the information passed to the callback function by PredictLlama.  The values in here are not
// valid after the callback returns -- information like Response should be copied out.
type Prediction struct {
	model *Model

	Model    string `json:"model"`
	Response []int  `json:"response"`
	Done     bool   `json:"done"`
}

// String implements llm.Prediction by decoding the predicted tokens.  This will be valid UTF-8.
func (r *Prediction) String() string { return r.model.Decode(r.Response...) }

// Tokens implements llm.Prediction by returning the predicted tokens, which is a series of tokens that decode to
// valid UTF-8.
func (r *Prediction) Tokens() []int { return r.Response }

// Context returns the context used to generate the prediction.
func (r *Prediction) Context() []int {
	src := r.model.embd
	embd := make([]int, len(src))
	for i := range src {
		embd[i] = int(src[i])
	}
	return embd
}

// Timings may be called to get the timings for the prediction.
func (r *Prediction) Timings() *PredictionTimings {
	timings := C.llama_get_timings(r.model.tokens)
	return &PredictionTimings{
		SampleCount:        int(timings.n_sample),
		SampleDuration:     parseDurationMs(float64(timings.t_sample_ms)),
		PromptEvalCount:    int(timings.n_p_eval),
		PromptEvalDuration: parseDurationMs(float64(timings.t_p_eval_ms)),
		EvalCount:          int(timings.n_eval),
		EvalDuration:       parseDurationMs(float64(timings.t_eval_ms)),
	}
}

type PredictionTimings struct {
	SampleCount        int
	SampleDuration     time.Duration
	PromptEvalCount    int
	PromptEvalDuration time.Duration
	EvalCount          int
	EvalDuration       time.Duration
}
