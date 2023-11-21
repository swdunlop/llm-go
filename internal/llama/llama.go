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

int llama_context_eval(struct llama_context *ctx, int pos, llama_token *tokens, int n_tokens) {
	if (n_tokens < 1) return 0;
	llama_batch batch = llama_batch_init(n_tokens, 0);
	batch.n_tokens = n_tokens;
	for (int i = 0; i < n_tokens; i++) {
		batch.token[i] = tokens[i];
		batch.pos[i] = pos + i;
		batch.seq_id[i] = 0;
		batch.logits[i] = false;
	}
	batch.logits[n_tokens - 1] = true;
	int e = llama_decode(ctx, batch);
	llama_batch_free(batch);
	return e;
}

llama_token llama_sample(
	struct llama_context *ctx,
	struct llama_sample_params *opts,
	int pos,
	llama_token *last_tokens, int n_last_tokens
) {
	float *logits = llama_get_logits_ith(ctx, pos);
	if (logits == NULL) {
		return 0;
	}
	int n_vocab = llama_n_vocab(llama_get_model(ctx));
	llama_token_data *data = malloc(sizeof(llama_token_data) * n_vocab);
	if (data == NULL) {
		return 0;
	}
	for (int i = 0; i < n_vocab; i++) {
		data[i].id = i;
		data[i].logit = logits[i];
		data[i].p = 0;
	}
	llama_token_data_array candidates = {data, n_vocab, false};

	if (n_last_tokens > 0) {
		llama_sample_repetition_penalty(
			ctx, &candidates,
			last_tokens, n_last_tokens,
			opts->repeat_penalty
		);

		llama_sample_frequency_and_presence_penalties(
			ctx, &candidates,
			last_tokens, n_last_tokens,
			opts->frequency_penalty, opts->presence_penalty
		);

		if (!opts->penalize_newline) {
			struct llama_token_data newline = data[llama_token_nl(ctx)];
			data[llama_token_nl(ctx)] = newline;
		}
	}

	llama_token token = 0;

	if (opts->temperature <= 0) {
		token = llama_sample_token_greedy(ctx, &candidates);
	} else if (opts->mirostat == 1) {
		int mirostat_m = 100;
		float mirostat_mu = 2.0f * opts->mirostat_tau;
		llama_sample_temp(ctx, &candidates, opts->temperature);
		token = llama_sample_token_mirostat(
			ctx, &candidates,
			opts->mirostat_tau, opts->mirostat_eta,
			mirostat_m, &mirostat_mu);
	} else if (opts->mirostat == 2) {
		float mirostat_mu = 2.0f * opts->mirostat_tau;
		llama_sample_temp(ctx, &candidates, opts->temperature);
		token = llama_sample_token_mirostat_v2(
			ctx, &candidates,
			opts->mirostat_tau, opts->mirostat_eta,
			&mirostat_mu);
	} else {
		llama_sample_top_k(ctx, &candidates, opts->top_k, 1);
		llama_sample_tail_free(ctx, &candidates, opts->tfs_z, 1);
		llama_sample_typical(ctx, &candidates, opts->typical_p, 1);
		llama_sample_top_p(ctx, &candidates, opts->top_p, 1);
		llama_sample_temp(ctx, &candidates, opts->temperature);
		token = llama_sample_token(ctx, &candidates);
	}

	free(data);
	return token;
}

void mute_log_handler(enum ggml_log_level level, const char *text, void *user) {
	(void)(user);

	// Only allow warnings, errors and things higher than INFO out.
	if (level <= GGML_LOG_LEVEL_INFO) return;
    fputs(text, stderr);
    fflush(stderr);
}

static void mute_llama_log() {
	llama_log_set(mute_log_handler, NULL);
}
*/
import "C"
import (
	"errors"
	"fmt"
	"io"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/rs/zerolog"
	"github.com/swdunlop/llm-go/internal/kmp"
)

func Load(modelPath string) (Model, error) {
	m := &model{}
	err := m.load(modelPath)
	if err != nil {
		return nil, err
	}
	return m, nil
}

type Model interface {
	Close()
	Encode(string) []Token
	Decode([]Token) string
	Predict(*zerolog.Logger, *Parameters, []Token) (Stream, error)
}

type Stream interface {
	Close()
	Next([]Token) (Token, error)
}

type model struct {
	llama        *C.struct_llama_model
	bos, eos, nl Token
	nCtx         int // trained context size
	nVocab       int // size of the model's vocabulary
	last         struct {
		control sync.Mutex
		stream  *stream
	}
}

func (m *model) load(modelPath string) error {
	// TODO: find a way to make this interruptible by context.
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	m.llama = C.llama_load_model_from_file(cPath, params.model)
	if m.llama == nil {
		return fmt.Errorf("failed to load %q", modelPath)
	}
	m.nCtx = int(C.llama_n_ctx_train(m.llama))
	if m.nCtx < 1 {
		C.llama_free_model(m.llama)
		return fmt.Errorf(`missing n_ctx_train in model %q`, modelPath)
	}
	m.nVocab = int(C.llama_n_vocab(m.llama))
	// TODO: This is.. Dumb.  We should be able to get this directly from the model's vocabulary.
	lcx := C.llama_new_context_with_model(m.llama, params.context)
	if lcx == nil {
		C.llama_free_model(m.llama)
		return fmt.Errorf("failed to create context from model %q", modelPath)
	}
	defer C.llama_free(lcx)
	m.bos = Token(C.llama_token_bos(lcx)) // TODO: update to use model.
	m.eos = Token(C.llama_token_eos(lcx)) // TODO: update to use model.
	m.nl = Token(C.llama_token_nl(lcx))   // TODO: update to use model.
	return nil
}

func (m *model) Close() {
	m.last.control.Lock()
	if m.last.stream != nil {
		m.last.stream.free()
		m.last.stream = nil
	}
	m.last.control.Unlock()
	if m.llama == nil {
		return
	}
	C.llama_free_model(m.llama)
	m.llama = nil
}

func (m *model) Encode(text string) []Token {
	// TODO: this can be done without copying.
	buf := make([]Token, len(text))
	n := C.llama_tokenize(
		m.llama,
		(*C.char)(unsafe.Pointer(unsafe.StringData(text))),
		C.int(len(text)),
		(*C.int)(unsafe.SliceData(buf)),
		C.int(len(buf)), false)
	if n < 0 {
		panic(errors.New(`llama_tokenize failed`))
	}
	return buf[:n]
}

func (m *model) Decode(tokens []Token) string {
	// TODO: This is a pointless memcpy, we may want to just look things up in the vocab directly.
	var buf strings.Builder
	var tmp [1024]byte // TODO: what is the right size?
	buf.Grow(len(tokens) * 4)
	// TODO: is this lock necessary?
	for _, token := range tokens {
		n := C.llama_token_to_piece(m.llama, C.int(token), (*C.char)(unsafe.Pointer(&tmp[0])), C.int(len(tmp)))
		if n < 0 {
			panic(fmt.Errorf(`llama_token_to_piece %v failed`, token))
		}
		buf.Write(tmp[:n])
	}
	str := buf.String()
	str = strings.TrimPrefix(str, ` `) // there is generally a leading space due to tokenization.
	return buf.String()
}

func (m *model) Predict(log *zerolog.Logger, pp *Parameters, tokens []Token) (Stream, error) {
	// note that we have to be careful not to lose it in the reset.
	var err error
	m.last.control.Lock()
	s := m.last.stream
	m.last.stream = nil
	m.last.control.Unlock()
	tokens = append([]Token{m.bos}, tokens...)
	if s == nil {
		s = &stream{model: m, log: log}
		err = s.init(tokens, pp)
	} else {
		s.log = log
		err = s.reset(tokens, pp)
	}
	if err != nil {
		s.free()
		return nil, err
	}
	return s, nil
}

type stream struct {
	log    *zerolog.Logger
	model  *model // note that this is a borrow of the model, and should not be closed by stream.
	llama  *C.struct_llama_context
	params struct {
		sample C.struct_llama_sample_params
	}

	history []Token
}

func (s *stream) init(tokens []Token, pp *Parameters) error {
	applyParameters(&s.params.sample, pp)

	cp := params.context
	cp.n_ctx = C.uint(s.model.nCtx)
	cp.n_batch = cp.n_ctx // TODO: support nBatch < nCtx; is nBatch even used?
	n := len(tokens)
	max := int(params.context.n_ctx - 5)
	if n > max {
		return fmt.Errorf("%v tokens of input exceeds maximum %v tokens", n, max)
	}

	s.history = make([]Token, 0, cp.n_ctx)
	s.log.Trace().
		Uint(`seed`, uint(pp.Seed)).
		Int(`nCtx`, s.model.nCtx).Int(`nBatch`, int(cp.n_batch)).
		Msg(`creating context`)
	s.llama = C.llama_new_context_with_model(s.model.llama, cp)
	if s.llama == nil {
		return fmt.Errorf("failed to create context from model")
	}
	C.llama_set_rng_seed(s.llama, C.uint32_t(pp.Seed))

	err := s.eval(tokens...)
	if err != nil {
		return err
	}
	return nil
}

// reset identifies the overlap between the history of the stream and the tokens then resets the batch.
func (s *stream) reset(tokens []Token, pp *Parameters) error {
	applyParameters(&s.params.sample, pp)

	n, m := len(tokens), cap(s.history)
	if n > m {
		return fmt.Errorf(`%v tokens of input exceeds maximum %v tokens`, len(tokens), cap(s.history))
	}
	sz, pos := kmp.Overlap(tokens, s.history)
	end := pos + sz
	s.log.Trace().Int(`history`, len(s.history)).Int(`pos`, pos).Int(`sz`, sz).Msg(`resetting stream`)

	rmCache(s.llama, 0, end, -1)
	shiftCache(s.llama, 0, pos, end, -pos)

	copy(s.history, s.history[pos:])
	s.history = s.history[:sz]
	err := s.eval(tokens[sz:]...)
	if err != nil {
		return err
	}
	return nil
}

// shiftCache moves the range of batch tokens from start to stop by delta.  seqID is generally 0 since we do not use
// batches concurrently.
func shiftCache(cache *C.struct_llama_context, seqID int, start, stop int, delta int) {
	if delta == 0 {
		return // no movement.
	}
	C.llama_kv_cache_seq_shift(cache, C.int(seqID), C.int(start), C.int(stop), C.int(delta))
}

// rmCache removes the range of batch tokens from start to stop.  If stop is -1, this removes all tokens from start.
// seqID is generally 0 since we do not use batches concurrently.
func rmCache(cache *C.struct_llama_context, seqID int, start, stop int) {
	C.llama_kv_cache_seq_rm(cache, C.int(seqID), C.int(start), C.int(stop))
}

func (s *stream) Close() {
	if s.llama == nil {
		return
	}
	s.model.last.control.Lock()
	if s.model.last.stream != nil {
		s.model.last.stream.free()
	}
	s.model.last.stream = s
	s.model.last.control.Unlock()
}

func (s *stream) free() {
	C.llama_free(s.llama)
	s.llama = nil
}

func (s *stream) Next(tokens []Token) (Token, error) {
	headspace := cap(s.history) - len(s.history) - 5
	if headspace < len(tokens) {
		return 0, ContextFull{}
	}
	// TODO: check for nBatch < nCtx
	err := s.eval(tokens...)
	if err != nil {
		return 0, err
	}
	token, err := s.sample()
	if err != nil {
		return 0, err
	}
	if token == s.model.eos {
		return 0, io.EOF
	}
	err = s.eval(token)
	if err != nil {
		return 0, err
	}
	return token, nil
}

func (s *stream) sample() (Token, error) {
	pos := len(s.history) - 1
	token := C.llama_sample(
		s.llama,
		&s.params.sample,
		C.int(pos),
		unsafe.SliceData(s.history),
		C.int(len(s.history)),
	)
	s.log.Trace().
		Int(`pos`, pos).
		Int(`token`, int(token)).
		Msg(`sampled token`)
	if token == 0 {
		return 0, errors.New(`llama failed to produce a token`)
	}
	return Token(token), nil
}

func (s *stream) eval(tokens ...Token) error {
	if len(tokens) == 0 {
		return nil
	}
	s.log.Trace().
		// Int(`history`, len(s.history)).
		Interface(`history`, s.history).
		Interface(`tokens`, tokens).
		Msg(`evaluating tokens`)
	e := C.llama_context_eval(s.llama, C.int(len(s.history)), unsafe.SliceData(tokens), C.int(len(tokens)))
	if e == 0 {
		s.history = append(s.history, tokens...)
		return nil
	}
	if e == 1 {
		return fmt.Errorf(`eval failed, cache overflow`)
	}
	return fmt.Errorf(`eval failed with error %v`, e)
}

type Token = C.int32_t

type ContextFull struct{}

func (ContextFull) Error() string { return `context full` }

var nThreads = runtime.NumCPU()

var params = struct {
	model    C.struct_llama_model_params
	quantize C.struct_llama_model_quantize_params
	context  C.struct_llama_context_params
}{
	C.llama_model_default_params(),
	C.llama_model_quantize_default_params(),
	C.llama_context_default_params(),
}

func applyParameters(opts *C.struct_llama_sample_params, pp *Parameters) {
	opts.temperature = C.float(pp.Temperature)
	opts.penalize_newline = C.bool(pp.PenalizeNL)
	opts.top_k = C.int(pp.TopK)
	opts.top_p = C.float(pp.TopP)
	opts.tfs_z = C.float(pp.TFSZ)
	opts.typical_p = C.float(pp.TypicalP)
	opts.repeat_penalty = C.float(pp.RepeatPenalty)
	// TODO opts.repeat_last_n = C.int(pp.RepeatLastN)
	opts.presence_penalty = C.float(pp.PresencePenalty)
	opts.frequency_penalty = C.float(pp.FrequencyPenalty)
	opts.mirostat = C.int(pp.Mirostat)
	opts.mirostat_tau = C.float(pp.MirostatTau)
	opts.mirostat_eta = C.float(pp.MirostatEta)
	if pp.Seed == 0 {
		for {
			u := uint32(rand.Int())
			if u != 0 {
				pp.Seed = u
				break
			}
		}
	}
}

type Parameters struct {
	// Temperature is the temperature to use for the prediction.  If 0, there will be no randomness in the sampling.
	Temperature float32 `json:"temperature,omitempty"`

	// PenalizeNL penalizes newline tokens when applying the repeat penalty (default: true).
	PenalizeNL bool `json:"penalize_nl"`

	// TopK limits the next token selection to the K most probable tokens, defaults to 40.
	TopK int `json:"top_k"`

	// TopP limits the next token selection to a subset of tokens with a cumulative probability above a threshold P, defaults to 0.9.
	TopP float32 `json:"top_p"`

	// NPredict specifies the number of tokens to predict when generating text. Note: May exceed the set limit slightly if the last token is a partial multibyte character. When 0, no tokens will be generated but the prompt is evaluated into the cache. (default: 128, -1 = infinity).
	NPredict int `json:"n_predict"`

	// NKeep specifies the number of tokens from the initial prompt to retain when the model resets its internal context. By default, this value is set to 0 (meaning no tokens are kept). Use -1 to retain all tokens from the initial prompt.
	// TODO NKeep int `json:"n_keep"`

	// Stop specifies a JSON array of stopping strings. These words will not be included in the completion, so make sure to add them to the prompt for the next iteration
	// TODO Stop []string `json:"stop,omitempty"`

	// TFSZ enables tail free sampling with parameter z (default: 1.0, 1.0 = disabled).
	TFSZ float32 `json:"tfsz"`

	// TypicalP enable locally typical sampling with parameter p (default: 1.0, 1.0 = disabled).
	TypicalP float32 `json:"typical_p"`

	// RepeatPenalty controls the repetition of token sequences in the generated text (default: 1.1).  This is ignored if RepeatLastN is 0.
	RepeatPenalty float32 `json:"repeat_penalty"`

	// RepeatLastN controls the number of tokens to consider for penalizing repetition (default: 64, 0 = disabled, -1 = ctx-size).  This is ignored if RepeatPenalty is 0.
	// TODO RepeatLastN int `json:"repeat_last_n"`

	// PresencePenalty specifies the repeat alpha presence penalty (default: 0.0, 0.0 = disabled).
	PresencePenalty float32 `json:"presence_penalty,omitempty"`

	// FrequencyPenalty specifies the repeat alpha frequency penalty (default: 0.0, 0.0 = disabled).
	FrequencyPenalty float32 `json:"frequency_penalty,omitempty"`

	// Mirostat enables Mirostat sampling, controlling perplexity during text generation (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
	Mirostat int `json:"mirostat"`

	// mirostat_tau: Set the Mirostat target entropy, parameter tau (default: 5.0).
	MirostatTau float32 `json:"mirostat_tau,omitempty"`

	// mirostat_eta: Set the Mirostat learning rate, parameter eta (default: 0.1).
	MirostatEta float32 `json:"mirostat_eta,omitempty"`

	// TODO: grammar
	// TODO: ignore_eos
	// TODO: logit_bias

	// Seed to use for prediction and sampling.  If 0, a random seed will be used.
	Seed uint32 `json:"seed,omitempty"`

	// Most of this comes from https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
	// Including the weird ass defaults and crappy names.   Compatibility with llama.cpp concepts is more important than
	// trying to fix those concepts (for now)
}

func Defaults() Parameters {
	return Parameters{
		TopK:          40,
		TopP:          0.9,
		NPredict:      128,
		TFSZ:          1.0,
		TypicalP:      1.0,
		RepeatPenalty: 1.1,
		MirostatTau:   5.0,
		MirostatEta:   0.1,
		Seed:          0,
	}
}

func init() {
	C.mute_llama_log()
}
