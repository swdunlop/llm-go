package llama

/*
#cgo CFLAGS: -Ofast -std=c11 -fPIC
#cgo CPPFLAGS: -Ofast -Wall -Wextra -Wno-unused-function -Wno-unused-variable -DNDEBUG -DGGML_USE_K_QUANTS
#cgo CXXFLAGS: -std=c++11 -fPIC
#cgo darwin CPPFLAGS:  -DGGML_USE_ACCELERATE
#cgo darwin,arm64 CPPFLAGS: -DGGML_USE_METAL -DGGML_METAL_NDEBUG
#cgo darwin LDFLAGS: -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
#include <stdlib.h>
#include "ggml-alloc.h"
#include "llama.h"

struct llama_sample_params
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
	struct llama_token_data *candidates, size_t n_candidates,
	const llama_token *last_tokens, size_t n_last_tokens,
	struct llama_sample_params *opts
) {
	llama_token_data_array candidates_p = {
		candidates,
		n_candidates,
		false,
	};

	if (n_last_tokens > 0) {
		struct llama_token_data newline = candidates_p.data[llama_token_nl(ctx)];

		llama_sample_repetition_penalty(
			ctx, &candidates_p,
			last_tokens, n_last_tokens,
			opts->repeat_penalty);

		llama_sample_frequency_and_presence_penalties(
			ctx, &candidates_p,
			last_tokens, n_last_tokens,
			opts->frequency_penalty, opts->presence_penalty);

		if (!opts->penalize_newline) {
			candidates_p.data[llama_token_nl(ctx)] = newline;
		}
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
	"context"
	"errors"
	"fmt"
	"math/rand"
	"runtime"
	"strings"
	"unsafe"

	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/llm-go/internal"
)

func init() {
	llm.Register(`llama`, func(cf map[string]any) (llm.Interface, error) {
		var model = struct {
			Model string `json:"model"`
			// TODO: Adapters
			NumCtx int `json:"ctx_size"`
		}{
			NumCtx: 4096,
		}
		err := llm.Unmap(cf, &model)
		if err != nil {
			return nil, err
		}
		predict := predictDefaults
		err = llm.Unmap(cf, &predict)
		if err != nil {
			return nil, err
		}
		err = llm.Unmap(cf, &predict)
		if err != nil {
			return nil, err
		}
		options := []Option{
			func(cfg *llama) { cfg.predictOptions = predict },
			ModelFile(model.Model),
			NCtx(model.NumCtx),
			Metal(),                        // TODO: make this configurable
			MLock(true),                    // TODO: make this configurable
			MMap(true),                     // TODO: make this configurable
			NThreads(runtime.NumCPU() - 2), // TODO: make this configurable
		}
		return New(options...)
	}, llmOptions...)
}

func New(options ...Option) (Interface, error) {
	var cfg llama
	cfg.modelParams = C.llama_context_default_params()
	for _, option := range options {
		option(&cfg)
		if cfg.err != nil {
			return nil, cfg.err
		}
	}
	err := cfg.init()
	if err != nil {
		return nil, err
	}
	return &cfg, nil
}

type Interface interface {
	llm.Interface

	// ContextSize returns the number of tokens in the model context.
	ContextSize() int

	// Encode tokenizes the given text using the model.
	Encode(text string) ([]Token, error)

	// Decode detokenizes the given tokens using the model.
	Decode(tokens ...Token) (string, error)

	// BOS returns the Beginning of Stream token for the model.
	BOS() Token

	// EOS returns the End of Stream token for the model.
	EOS() Token

	// Release releases the resources associated with the model.
	Release()
}

// Metal enables Metal acceleration for the model, if running on a Mac.
func Metal() Option {
	return modelOption(func(p *modelParams) {
		if runtime.GOOS == "darwin" {
			p.n_gpu_layers = 1
		}
	})
}

// NThreads specifies the number of threads to use for evaluation, default is 1.
func NThreads(n int) Option { return func(l *llama) { l.nThreads = n } }

// NCtx specifies the number of tokens in the model context, default is 512 but most LLaMA models support 2048 or
// more as of August, 2023.
func NCtx(n int) Option { return modelOption(func(p *modelParams) { p.n_ctx = C.int32_t(n) }) }

// ModelFile specifies the path to the model file which contains the model weights and other metadata.
func ModelFile(path string) Option { return func(c *llama) { c.modelFile = path } }

// MLock locks the model weights in memory, which can improve performance.
func MLock(ok bool) Option { return modelOption(func(p *modelParams) { p.use_mlock = C.bool(ok) }) }

// MMap maps the model file into memory instead of reading it, which may improve load speed.
func MMap(ok bool) Option { return modelOption(func(p *modelParams) { p.use_mmap = C.bool(ok) }) }

// Temperature specifies the sampling temperature, default is 1.0.  Higher temperatures generate more random samples,
// lower temperatures generate more deterministic samples.  If temperature is 0 or lower, greedy sampling is used
// instead, with the most probable token always being selected.
func Temperature(t float32) PredictOption {
	return func(opts *predictOptions) { opts.Temperature = t }
}

func modelOption(fn func(p *modelParams)) Option {
	return func(cfg *llama) {
		if cfg.initialized {
			// TODO: support re-starting the context with new options
			return // ignore model options after initialization
		}
		fn(&cfg.modelParams)
	}
}

type Option func(*llama)

type llama struct {
	initialized    bool
	modelParams    modelParams
	modelFile      string
	nThreads       int
	predictOptions predictOptions
	err            error

	model   *C.struct_llama_model
	llama   *C.struct_llama_context
	history []Token
}

type sampleParams = C.struct_llama_sample_params

func (cfg *llama) init() error {
	cfg.initialized = true
	cfg.model = C.llama_load_model_from_file(C.CString(cfg.modelFile), cfg.modelParams)
	if cfg.model == nil {
		return fmt.Errorf("failed to load %q", cfg.modelFile)
	}
	cfg.llama = C.llama_new_context_with_model(cfg.model, cfg.modelParams)
	if cfg.llama == nil {
		return fmt.Errorf("failed to create context from model %q", cfg.modelFile)
	}
	boot := []Token{cfg.BOS()}
	cfg.seed(42)
	return cfg.eval(boot, 0)
}

// Encode tokenizes the given text using the model.
func (cfg *llama) Encode(text string) ([]Token, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	buf := make([]Token, len(text))
	// TODO: should we consider support BOS?
	n := C.llama_tokenize(cfg.llama, ctext, unsafe.SliceData(buf), C.int(len(buf)), false)
	if n < 0 {
		return nil, errors.New(`llama_tokenize failed`)
	}
	return buf[:n], nil
}

// Decode detokenizes the given tokens using the model.
func (cfg *llama) Decode(tokens ...Token) (string, error) {
	// TODO: This is a pointless memcpy, we may want to just look things up in the vocab directly.
	var buf strings.Builder
	var tmp [1024]byte // TODO: what is the right size?
	buf.Grow(len(tokens) * 4)
	// TODO: is this lock necessary?
	for _, token := range tokens {
		n := C.llama_token_to_piece(cfg.llama, token, (*C.char)(unsafe.Pointer(&tmp[0])), C.int(len(tmp)))
		if n < 0 {
			return buf.String(), fmt.Errorf(`llama_token_to_piece %v failed`, token)
		}
		buf.Write(tmp[:n])
	}
	str := buf.String()
	str = strings.TrimPrefix(str, ` `) // there is generally a leading space due to tokenization.
	return buf.String(), nil
}

// Predict implements the llm.Predictor interface.
func (cfg *llama) Predict(ctx context.Context, cf map[string]any, content []string, fn func(llm.Prediction) error) (string, error) {
	// json.NewEncoder(os.Stderr).Encode(cf) // println
	return cfg.PredictLlama(ctx, content,
		PredictConfiguration(cf),
		Callback(func(p *Prediction) error { return fn(p) }),
	)
}

// PredictLlama returns the most likely tokens for the given context.
func (cfg *llama) PredictLlama(ctx context.Context, content []string, options ...PredictOption) (out string, err error) {
	opts := cfg.predictOptions
	for _, opt := range options {
		opt(&opts)
		if opts.err != nil {
			return ``, opts.err
		}
	}
	var params sampleParams
	opts.setParams(&params)

	instructions, err := cfg.Encode(opts.Instructions)
	if err != nil {
		return ``, err
	}
	filter := internal.NewStopFilter(opts.Stop...)
	keep := len(instructions)
	max := (int(cfg.modelParams.n_ctx) - keep) * 2 / 3 // 66% of non-instruction context is available for content.

	contentTokens, err := cfg.inputTokens(content)
	if err != nil {
		return ``, err
	}

	cfg.seed(opts.Seed)

reset:
	tokens := compact(max, instructions, contentTokens)
	if tokens == nil {
		return ``, fmt.Errorf(`content and instructions overflow context size %v`, cfg.modelParams.n_ctx)
	}

	past := cfg.pastBatch(tokens)
	history := make([]Token, past, cfg.modelParams.n_ctx)
	copy(history, cfg.history) // TODO: this does not handle a reset well.
	defer func() {
		// println("resulting history", len(history))
		cfg.history = history
	}()

	var buf strings.Builder
	buf.Grow(16384)
	defer func() {
		buf.WriteString(filter.String())
		out = buf.String()
	}()

	eos := cfg.EOS()
	for {
		var text string
		if err = ctx.Err(); err != nil {
			return
		}
		// text, _ := cfg.Decode(tokens...)
		// fmt.Printf("predicting %v tokens, past %v, n_ctx %v\n", len(tokens), past, cfg.modelParams.n_ctx)
		err = cfg.eval(tokens, past)
		if errors.Is(err, errPastOverflow{}) {
			// fmt.Printf("%v tokens, past overflow, resetting\n", len(tokens))
			cfg.history = history
			contentTokens = append(contentTokens, tokens)
			goto reset
		}
		if err != nil {
			return
		}

		history = append(history, tokens...)
		past += len(tokens)
		var token Token
		token, err = cfg.sample(&params, nil) // TODO: provide rest.
		if err != nil {
			return
		}
		if token == eos {
			return
		}
		tokens = []Token{token}
		// var text string
		text, err = cfg.Decode(token)
		if err != nil {
			return
		}
		// fmt.Printf("filtering %q\n", text)
		text, stop := filter.Filter(text)
		if text != `` {
			buf.WriteString(text)
			if opts.callback != nil {
				err = opts.callback(&Prediction{Output: text})
				if err != nil {
					// TODO: output the buffer, not just the filter.
					return
				}
			}
		}
		// TODO: look for EOS token
		if stop {
			return
		}
	}
}

// pastBatch compares the batch to the history to determine the nPast
func (cfg *llama) pastBatch(batch []Token) int {
	if len(batch) > len(cfg.history) {
		batch = batch[:len(cfg.history)]
	}
	for i, t := range batch {
		if cfg.history[i] != t {
			return i
		}
	}
	return len(batch)
}

func (cfg *llama) seed(u uint32) {
	if u == 0 {
		u = rand.Uint32()
	}
	C.llama_set_rng_seed(cfg.llama, C.uint32_t(u))
}

func (cfg *llama) inputTokens(inputs []string) ([][]Token, error) {
	ret := make([][]Token, len(inputs))
	for i, input := range inputs {
		var err error
		ret[i], err = cfg.Encode(input)
		if err != nil {
			return nil, err
		}
	}
	return ret, nil
}

// eval evaluates the given input, calculating npast using history.
func (cfg *llama) eval(batch []Token, nPast int) error {
	if nPast+len(batch) >= int(cfg.modelParams.n_ctx) {
		return fmt.Errorf(`%w %v`, errPastOverflow{}, cfg.modelParams.n_ctx)
	}
	// fmt.Fprintf(os.Stderr, "evaluating %v tokens, nPast %v\n", len(batch), nPast)
	C.llama_reset_timings(cfg.llama)
	// if we give more than 256 tokens to eval in one go, we encounter buffer allocation issues in llama.cpp
	// as of 2023-09-01.  This was not true before GGUF.
	for len(batch) > 0 {
		f := 256
		if f > len(batch) {
			f = len(batch)
		}
		res := C.llama_eval(cfg.llama, unsafe.SliceData(batch), C.int(f), C.int(nPast), C.int(cfg.nThreads))
		if res != 0 {
			return errors.New("llama_eval failed")
		}
		batch = batch[f:]
		nPast += f
	}
	return nil
}

type errPastOverflow struct{}

func (errPastOverflow) Error() string { return `past overflows context limit` }

// Sample selects the most likely token from the model after Eval.  The recent slice is used by the "recent" options
// to penalize tokens that have recently been selected.
func (cfg *llama) Sample(options ...PredictOption) (Token, error) {
	opts := cfg.predictOptions
	for _, opt := range options {
		opt(&opts)
		if opts.err != nil {
			return 0, opts.err
		}
	}
	cfg.seed(opts.Seed)
	var params sampleParams
	opts.setParams(&params)
	return cfg.sample(&params, opts.recent)
}

func (cfg *llama) sample(params *sampleParams, recent []Token) (Token, error) {
	// TODO: this is a /large/ allocation, n_vocab is 32000 for llama2
	numVocab := C.llama_n_vocab(cfg.llama)
	logits := unsafe.Slice(C.llama_get_logits(cfg.llama), numVocab)
	candidates := make([]C.llama_token_data, numVocab)
	for i := range logits {
		candidates[i] = C.llama_token_data{
			id:    C.int(i),
			logit: logits[i],
			p:     0,
		}
	}

	var recentPtr *Token
	var recentLen C.size_t
	if recent != nil {
		recentPtr = unsafe.SliceData(recent)
		recentLen = C.size_t(len(recent))
	}

	return Token(C.llama_sample(
		cfg.llama,
		unsafe.SliceData(candidates),
		C.size_t(len(candidates)),
		recentPtr, recentLen,
		params,
	)), nil
}

// PredictOptions are options for Predict and Sample.
func (cfg *llama) PredictOptions() predictOptions { return cfg.predictOptions }

// BOS returns the Beginning of Stream token for the model.
func (cfg *llama) BOS() Token { return C.llama_token_bos(cfg.llama) }

// EOS returns the End of Stream token for the model.
func (cfg *llama) EOS() Token { return C.llama_token_eos(cfg.llama) }

func (cfg *llama) Release() {
	if cfg.llama != nil {
		C.llama_free(cfg.llama)
		cfg.llama = nil
	}
	// TODO: split the model, context and sampling apart into their own types so they can be shared and released
	// properly.
	if cfg.model != nil {
		C.llama_free_model(cfg.model)
		cfg.model = nil
	}
}

func (cfg *llama) ContextSize() int { return int(cfg.modelParams.n_ctx) }

type modelParams = C.struct_llama_context_params

// compact will use the provided encoder to encode the combination of prefix, content and suffix, producing a slice of
// tokens that has the following properties:
//
//  1. The number of tokens is less than or equal to max.  If this is not possible, nil is returned.
//  2. All of the tokens from the prefix and suffix are included.
//  3. As many tokens as possible from the content are included.
//
// This is done by dropping lines of text from the front of the content until it fits.  Some assumptions are made, here:
// that text is peppered with newlines, that newlines have a single unambiguous encoding, and that the encoder will
// produce the same encoding for the same input.  This is true for LLaMA encoders.
func compact(max int, instructions []Token, contents [][]Token) []Token {
	n := len(instructions)
	max -= len(instructions)
	if max < 0 {
		return nil // not enough space for prefix and suffix
	}
	contentSize := 0
	for _, content := range contents {
		contentSize += len(content)
	}
	buf := make([]Token, n, max)
	copy(buf, instructions)
	for {
		if contentSize <= max {
			for _, content := range contents {
				buf = append(buf, content...)
			}
			return buf
		}
		if len(contents) == 1 {
			// This is desperate, since the start of the content is likely to be a fragmented token or the content
			// itself is fragmented.
			content := contents[0]
			return append(buf, content[contentSize-max:]...)
		}
		contentSize -= len(contents[0])
		contents = contents[1:]
	}
}

// A Token is a single token understood by a LLaMA model.
type Token = C.int32_t

// Sample adds sampling options that modify every call to Sample.
func Sample(options ...PredictOption) Option {
	return func(cfg *llama) {
		for _, option := range options {
			option(&cfg.predictOptions)
			if cfg.predictOptions.err != nil {
				cfg.err = cfg.predictOptions.err
				return
			}
		}
	}
}

// Predict adds options that modify every call to Predict.
func Predict(options ...PredictOption) Option {
	return func(cfg *llama) {
		for _, option := range options {
			option(&cfg.predictOptions)
			if cfg.predictOptions.err != nil {
				cfg.err = cfg.predictOptions.err
				return
			}
		}
	}
}

// Stop replaces the set of stop tokens used by Predict.
func Stop(stop ...string) PredictOption {
	return func(opts *predictOptions) {
		opts.Stop = stop
	}
}

func PredictConfiguration(cf map[string]any) PredictOption {
	return func(opts *predictOptions) {
		opts.err = llm.Unmap(cf, opts)
	}
}

// Callback sets a callback that will be called every time predict outputs tokens that cannot involve a stop token.
func Callback(fn func(*Prediction) error) PredictOption {
	return func(opts *predictOptions) {
		opts.callback = fn
	}
}

// A PredictOption modifies the behavior of Predict.
type PredictOption func(*predictOptions)

// predictOptions controls the behavior of Predict.
type predictOptions struct {
	RepeatPenalty    float32 `json:"repeat_penalty"`
	FrequencyPenalty float32 `json:"frequency_penalty"`
	PresencePenalty  float32 `json:"presence_penalty"`
	Temperature      float32 `json:"temperature"`
	TopK             int     `json:"top_k"`
	TopP             float32 `json:"top_p"`
	TFSZ             float32 `json:"tfs_z"`
	TypicalP         float32 `json:"typical_p"`
	Mirostat         int     `json:"mirostat"`
	MirostatTau      float32 `json:"mirostat_tau"`
	MirostatEta      float32 `json:"mirostat_eta"`
	PenalizeNewline  bool    `json:"penalize_newline"`
	Seed             uint32  `json:"seed"` // Used to seed both predict and sample.

	// Instruction is a string that will always be in context when predicting.  This is typically used for things like
	// instructions to the model that should always be present.
	//
	// The more instructions you have, the less of the context will be used for recent input, so use this sparingly.
	Instructions string `json:"instructions"`

	// Stop is a list of strings that, if they occur in the output, will cause prediction to stop and they will be
	// omitted from the output.  This is typically used for things like reverse prompts.
	Stop []string `json:"stop"`

	callback func(*Prediction) error
	recent   []Token `json:"-"`
	err      error   `json:"-"`
}

// Recent provides Sample with a slice of recent tokens for penalizing recently selected tokens.
func Recent(tokens []Token) func(*predictOptions) {
	return func(opts *predictOptions) {
		opts.recent = tokens
	}
}

var llmOptions = []llm.Option{
	// Model Options
	{Name: `model`, Value: ``, Use: `Model file name.`, Init: true},
	{Name: `ctx_size`, Value: 4096, Use: `Number of tokens in the model context.`, Init: true},
	// ngl, mg, ts, lv, mlock, nmap, numa are controlled by the package and not options.
	// TODO: lora

	// Sample / Predict Options
	{Name: `repeat_penalty`, Value: 1.1, Use: `Adjust the repetition penalty. 1.0 means no penalty. 1.2 means +20% chance of repeating. 0.8 means -20% chance of repeating.`},
	{Name: `temperature`, Value: 0.8, Use: `Adjust the randomness of the generated text.`},
	{Name: `top_k`, Value: 40, Use: `Limit the next token selection to the K most probable tokens.`},
	{Name: `top_p`, Value: 0.9, Use: `Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.`},
	{Name: `tfs_z`, Value: 1.0, Use: `Adjust the tail free sampling parameter.`},
	{Name: `typical_p`, Value: 1.0, Use: `Adjust the typicality sampling parameter.`},
	{Name: `mirostat`, Value: 0, Use: `Enable mirostat sampling; 0 = off, 1 = mirostat, 2 = mirostat v2.`},
	{Name: `mirostat_tau`, Value: 5.0, Use: `Set the Mirostat target entropy, parameter tau.  Higher values mean more random samples. Ignored if mirostat = 0.`},
	{Name: `mirostat_eta`, Value: 0.1, Use: `Set the Mirostat learning rate, parameter eta.  Higher values mean faster learning. Ignored if mirostat = 0.`},
	{Name: `penalize_newline`, Value: true, Use: `Penalize newline tokens in sampling.`},

	{Name: `seed`, Value: 0, Use: `Sets the random seed used in sampling.  0 means generate a new seed each time.`},
}

var predictDefaults = predictOptions{
	Instructions:     "",
	Stop:             nil,
	callback:         nil,
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

func (opts *predictOptions) setParams(ref *C.struct_llama_sample_params) {
	ref.repeat_penalty = C.float(opts.RepeatPenalty)
	ref.frequency_penalty = C.float(opts.FrequencyPenalty)
	ref.presence_penalty = C.float(opts.PresencePenalty)
	ref.temperature = C.float(opts.Temperature)
	ref.top_k = C.int(opts.TopK)
	ref.top_p = C.float(opts.TopP)
	ref.tfs_z = C.float(opts.TFSZ)
	ref.typical_p = C.float(opts.TypicalP)
	ref.mirostat = C.int(opts.Mirostat)
	ref.mirostat_tau = C.float(opts.MirostatTau)
	ref.mirostat_eta = C.float(opts.MirostatEta)
	ref.penalize_newline = C.bool(opts.PenalizeNewline)
}

// Predictions are produced in a stream by Predict and passed to a callback.
type Prediction struct {
	Output string `json:"output"`
}

func (p *Prediction) String() string {
	return p.Output
}
