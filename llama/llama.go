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
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
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
			`use_numa`: {`true`},
		}}
		var useNuma bool
		err := configuration.Get(&useNuma, cf, `use_numa`)
		if err != nil {
			initError = err
			return
		}
		C.llama_backend_init(C.bool(useNuma))
		defaultParams = C.llama_context_default_params()
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

func init() { llm.RegisterPredictor(`llama`, New) }

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

type llama struct {
	params *C.struct_llama_context_params
	model  *C.struct_llama_model
	tokens *C.struct_llama_context

	last   []C.llama_token
	embd   []C.llama_token
	cursor int

	mu sync.Mutex
	gc bool

	Options
}

type Options struct {
	Model    string   `json:"model,omitempty"`
	Adapters []string `json:"adapters,omitempty"`

	Seed int `json:"seed,omitempty"`

	// Backend options
	UseNUMA bool `json:"numa,omitempty"`

	// Model options
	NumCtx             int     `json:"n_ctx,omitempty"`
	NumBatch           int     `json:"n_batch,omitempty"`
	NumGQA             int     `json:"n_gqa,omitempty"`
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

	// Predict options
	NumKeep          int      `json:"num_keep,omitempty"`
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

	NumThread int `json:"n_threads,omitempty"`
}

// RuntimeDefault provides a default configuration for llama based on the runtime environment.
func RuntimeDefault() configuration.Interface {
	cf := Default()
	if runtime.GOOS == `darwin` && runtime.GOARCH == `arm64` {
		// we only want 1 thread per performance core on Apple Silicon, leave the efficiency cores alone

	}
	cf[`n_threads`] = []string{strconv.Itoa(ncpu)}
	return cf
}

// Default provides a default configuration for llama that omits runtime-specific settings.
func Default() configuration.Map {
	f := func(v any) []string { return []string{fmt.Sprint(v)} }
	return configuration.Map{
		`seed`:         {`-1`}, // Not f(defaultParams.seed),
		`n_ctx`:        f(defaultParams.n_ctx),
		`n_batch`:      f(defaultParams.n_batch),
		`n_gpu_layers`: f(defaultParams.n_gpu_layers),
		`n_gqa`:        f(defaultParams.n_gqa),
		// TODO: add rms_norm_eps
		// TODO: add tensor_split
		`low_vram`: f(defaultParams.low_vram),
		// TODO: add mul_mat_q
		`f16kv`: f(defaultParams.f16_kv),
		// TODO: add logits_all
		`rope_freq_base`:  f(defaultParams.rope_freq_base),
		`rope_freq_scale`: f(defaultParams.rope_freq_scale),
		// TODO: add vocab_only
		`use_mmap`:       f(defaultParams.use_mmap),
		`use_mlock`:      f(defaultParams.use_mlock),
		`embedding_only`: f(defaultParams.embedding), // Not f(defaultParams.embedding_only), ?

		`num_keep`:          {`-1`},   // TODO(swdunlop): move to generation options
		`repeat_last_n`:     {`64`},   // TODO(swdunlop): move to generation options
		`repeat_penalty`:    {`1.1`},  // TODO(swdunlop): move to generation options
		`frequency_penalty`: {`0.0`},  // TODO(swdunlop): move to generation options
		`presence_penalty`:  {`0.0`},  // TODO(swdunlop): move to generation options
		`temperature`:       {`0.8`},  // TODO(swdunlop): move to generation options
		`top_k`:             {`40`},   // TODO(swdunlop): move to generation options
		`top_p`:             {`0.9`},  // TODO(swdunlop): move to generation options
		`tfs_z`:             {`1.0`},  // TODO(swdunlop): move to generation options
		`typical_p`:         {`1.0`},  // TODO(swdunlop): move to generation options
		`mirostat`:          {`0`},    // TODO(swdunlop): move to generation options
		`mirostat_tau`:      {`5.0`},  // TODO(swdunlop): move to generation options
		`mirostat_eta`:      {`0.1`},  // TODO(swdunlop): move to generation options
		`penalize_newline`:  {`true`}, // TODO(swdunlop): move to generation options
	}
}

// New creates a new llm.Predictor from a configuration.
func New(cf configuration.Interface) (llm.Predictor, error) {
	err := Init(cf)
	if err != nil {
		return nil, err
	}
	llm := new(llama)

	cf = configuration.Overlay{cf, RuntimeDefault()}
	err = configuration.Unmarshal(&llm.Options, cf)
	if err != nil {
		return nil, err
	}

	// TODO(swdunlop): split prediction and model options apart.
	if _, err := os.Stat(llm.Options.Model); err != nil {
		return nil, err
	}

	f, err := os.Open(llm.Options.Model)
	if err != nil {
		return nil, err
	}
	defer f.Close() // TODO(swdunlop): we really don't need the file open the whole time we load this.

	//TODO(swdunlop): do we need decodeGGML at all? shouldn't we just use the llama.cpp API for this?
	//TODO(swdunlop): if we do, we should move this to a separate package and add gguf as well.
	//TODO(swdunlop): if we don't, how do we load older GGML models? do we have to offer a converter?
	ggml, err := decodeGGML(f, modelFamilyLlama)
	if err != nil {
		return nil, err
	}

	switch ggml.FileType().String() {
	case "F32", "F16", "Q5_0", "Q5_1", "Q8_0":
		if llm.Options.NumGPU != 0 {
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

	// TODO(swdunlop): this needs to be done once based on OS environment.  We can't flip this on each model.
	C.llama_backend_init(C.bool(llm.UseNUMA))

	params := C.llama_context_default_params()
	params.seed = C.uint(llm.Seed)
	params.n_ctx = C.int(llm.NumCtx)
	params.n_batch = C.int(llm.NumBatch)
	params.n_gqa = C.int(llm.NumGQA)
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

	if len(llm.Options.Adapters) > 0 && llm.UseMMap {
		return nil, fmt.Errorf(`you cannot combine mmap with lora adapters`)
	}

	llm.params = &params

	cModel := C.CString(llm.Options.Model)
	defer C.free(unsafe.Pointer(cModel))

	llm.model = C.llama_load_model_from_file(cModel, params)
	if llm.model == nil {
		return nil, errors.New("failed to load model")
	}

	llm.tokens = C.llama_new_context_with_model(llm.model, params)
	if llm.tokens == nil {
		return nil, errors.New("failed to create context")
	}

	for _, adapter := range llm.Options.Adapters {
		cAdapter := C.CString(adapter)
		defer C.free(unsafe.Pointer(cAdapter))

		if retval := C.llama_model_apply_lora_from_file(llm.model, cAdapter, nil, C.int(llm.NumThread)); retval != 0 {
			return nil, fmt.Errorf("failed to load adapter %s", adapter)
		}
	}

	// warm up the model
	bos := []C.llama_token{C.llama_token_bos()}
	C.llama_eval(llm.tokens, unsafe.SliceData(bos), C.int(len(bos)), 0, C.int(llm.Options.NumThread))
	C.llama_reset_timings(llm.tokens)

	return llm, nil
}

// Interface describes the full interface produced by New.
type Interface interface {
	llm.Predictor

	PredictLlama(context.Context, []int, func(*Prediction) error) error
	Embedding(string) ([]float64, error)
	Encode(string) []int
	Decode(...int) string
	SetOptions(Options) // TODO(swdunlop): move Options to a map[string]any
	Release()
}

func (llm *llama) Release() {
	llm.gc = true

	llm.mu.Lock()
	defer llm.mu.Unlock()

	defer C.llama_free_model(llm.model)
	defer C.llama_free(llm.tokens)

	C.llama_print_timings(llm.tokens)
}

func (llm *llama) SetOptions(opts Options) {
	// TODO(swdunlop): remove / deprecate, we should have a With method that provides a Predictor with prediction
	// options.
	llm.Options = opts
}

var errNeedMoreData = errors.New("need more data")

func (m *llama) Predict(ctx context.Context, content string, fn func(llm.Prediction) error) (string, error) {
	var buf strings.Builder
	err := m.PredictLlama(ctx, m.Encode(content), func(p *Prediction) error {
		buf.WriteString(p.Response)
		err := fn(p)
		return err
	})
	return buf.String(), err
}

func (m *llama) PredictLlama(ctx context.Context, tokens []int, fn func(*Prediction) error) error {
	C.llama_reset_timings(m.tokens)

	m.marshalPrompt(tokens)

	C.llama_set_rng_seed(m.tokens, C.uint(m.Seed))

	var b bytes.Buffer
	for {
		token, err := m.next(ctx)
		if m.gc {
			return nil
		} else if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return err
		}

		b.WriteString(m.Decode(int(token)))

		if err := m.checkStopConditions(b); err != nil {
			if errors.Is(err, io.EOF) {
				break
			} else if errors.Is(err, errNeedMoreData) {
				continue
			}

			return err
		}

		if utf8.Valid(b.Bytes()) || b.Len() >= utf8.UTFMax {
			err := fn(&Prediction{Response: b.String()})
			if err != nil {
				return err
			}
			b.Reset()
		}
	}

	embd := make([]int, len(m.embd))
	for i := range m.embd {
		embd[i] = int(m.embd[i])
	}

	timings := C.llama_get_timings(m.tokens)
	return fn(&Prediction{
		Done:               true,
		Context:            embd,
		SampleCount:        int(timings.n_sample),
		SampleDuration:     parseDurationMs(float64(timings.t_sample_ms)),
		PromptEvalCount:    int(timings.n_p_eval),
		PromptEvalDuration: parseDurationMs(float64(timings.t_p_eval_ms)),
		EvalCount:          int(timings.n_eval),
		EvalDuration:       parseDurationMs(float64(timings.t_eval_ms)),
	})
}

func (llm *llama) checkStopConditions(b bytes.Buffer) error {
	for _, stopCondition := range llm.Stop {
		if stopCondition == strings.TrimSpace(b.String()) {
			return io.EOF
		} else if strings.HasPrefix(stopCondition, strings.TrimSpace(b.String())) {
			return errNeedMoreData
		}
	}

	return nil
}

func (llm *llama) marshalPrompt(ctx []int) []C.llama_token {
	if llm.NumKeep < 0 {
		llm.NumKeep = len(ctx)
	}

	cTokens := make([]C.llama_token, len(ctx))
	for i := range ctx {
		cTokens[i] = C.llama_token(ctx[i])
	}

	// min(llm.NumCtx - 4, llm.NumKeep)
	if llm.NumCtx-4 < llm.NumKeep {
		llm.NumKeep = llm.NumCtx - 4
	}

	if len(ctx) >= llm.NumCtx {
		// truncate input
		numLeft := (llm.NumCtx - llm.NumKeep) / 2
		truncated := cTokens[:llm.NumKeep]
		erasedBlocks := (len(cTokens) - llm.NumKeep - numLeft - 1) / numLeft
		truncated = append(truncated, cTokens[llm.NumKeep+erasedBlocks*numLeft:]...)
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

func (llm *llama) Encode(prompt string) []int {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cTokens := make([]C.llama_token, len(prompt)+1)
	if n := C.llama_tokenize(llm.tokens, cPrompt, unsafe.SliceData(cTokens), C.int(len(cTokens)), true); n > 0 {
		tokens := make([]int, n)
		for i := range cTokens[:n] {
			tokens[i] = int(cTokens[i])
		}

		return tokens
	}

	return nil
}

func (llm *llama) Decode(tokens ...int) string {
	var sb strings.Builder
	for _, token := range tokens {
		sb.WriteString(C.GoString(C.llama_token_to_str(llm.tokens, C.llama_token(token))))
	}

	return sb.String()
}

func (llm *llama) next(ctx context.Context) (C.llama_token, error) {
	llm.mu.Lock()
	defer llm.mu.Unlock()

	if len(llm.embd) >= llm.NumCtx {
		numLeft := (llm.NumCtx - llm.NumKeep) / 2
		truncated := llm.embd[:llm.NumKeep]
		truncated = append(truncated, llm.embd[len(llm.embd)-numLeft:]...)

		llm.embd = truncated
		llm.cursor = llm.NumKeep
		log.Printf("input truncated: num_ctx=%d num_keep=%d num_left=%d num_tokens=%d cursor=%d", llm.NumCtx, llm.NumKeep, numLeft, len(truncated), llm.cursor)
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

	var sampleOpts C.struct_llama_sample_options
	sampleOpts.repeat_penalty = C.float(llm.RepeatPenalty)
	sampleOpts.frequency_penalty = C.float(llm.FrequencyPenalty)
	sampleOpts.presence_penalty = C.float(llm.PresencePenalty)
	sampleOpts.temperature = C.float(llm.Temperature)
	sampleOpts.top_k = C.int(llm.TopK)
	sampleOpts.top_p = C.float(llm.TopP)
	sampleOpts.tfs_z = C.float(llm.TFSZ)
	sampleOpts.typical_p = C.float(llm.TypicalP)
	sampleOpts.mirostat = C.int(llm.Mirostat)
	sampleOpts.mirostat_tau = C.float(llm.MirostatTau)
	sampleOpts.mirostat_eta = C.float(llm.MirostatEta)
	sampleOpts.penalize_newline = C.bool(llm.PenalizeNewline)

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

	repeatLastN := llm.RepeatLastN
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

func (llm *llama) Embedding(input string) ([]float64, error) {
	if !llm.EmbeddingOnly {
		return nil, errors.New("llama: embedding not enabled")
	}

	tokens := llm.Encode(input)
	if tokens == nil {
		return nil, errors.New("llama: tokenize embedding")
	}

	cTokens := make([]C.llama_token, len(tokens))
	for i := range tokens {
		cTokens[i] = C.llama_token(tokens[i])
	}

	retval := C.llama_eval(llm.tokens, unsafe.SliceData(cTokens), C.int(len(tokens)), 0, C.int(llm.NumThread))
	if retval != 0 {
		return nil, errors.New("llama: eval")
	}

	C.llama_print_timings(llm.tokens)

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

func (r *Prediction) String() string { return r.Response }
func (r *Prediction) Tokens() []int  { return r.Context }

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
