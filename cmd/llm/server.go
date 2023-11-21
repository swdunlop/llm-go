package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/gorilla/websocket"
	"github.com/rs/zerolog"
	"github.com/swdunlop/html-go/hog"
	"github.com/swdunlop/llm-go"
	"github.com/swdunlop/llm-go/predict"
	"github.com/swdunlop/zugzug-go"
	"github.com/swdunlop/zugzug-go/zug/parser"
)

var (
	cfgListenAddr   = `localhost:7272`
	cfgServeRoot    = ``
	cfgModelRoot    = `.`
	cfgDefaultModel = `mistral-7b-v0.1.Q5_K_M.gguf`
)

func init() {
	tasks = append(tasks, zugzug.Tasks{
		{Name: `service`, Fn: runServer, Use: `runs a prediction server`, Parser: parser.New(
			parser.String(&cfgListenAddr, `listen`, `l`, `address to listen on`),
			parser.String(&cfgServeRoot, `www`, `w`, `directory of WWW assets to serve; leave empty to disable`),
			parser.String(&cfgModelRoot, `models`, `m`, `directory of LLM models allowed by users`),
			parser.String(&cfgDefaultModel, `default-model`, `d`, `default model for users who do not specify one`),
		)},
	}...)
}

func runServer(ctx context.Context) error {
	svc := service{www: cfgServeRoot, defaultModel: cfgDefaultModel}
	svc.websocket.upgrader = websocket.Upgrader{
		HandshakeTimeout:  time.Second * 5,
		ReadBufferSize:    1024,    // 1 KiB // we expand this after the handshake
		WriteBufferSize:   4 << 20, // 4 MiB
		EnableCompression: false,   // Safari has recent problems with compression; let HTTP/2 handle it.
	}
	svc.logger = zerolog.Ctx(ctx)

	svr := http.Server{
		BaseContext: func(_ net.Listener) context.Context { return ctx },
		Handler:     svc.handler(),
	}
	go func(ctx context.Context) {
		<-ctx.Done()
		svc.logger.Info().Msg(`shutting down service`)
		shutdownCtx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()
		svr.Shutdown(shutdownCtx)
	}(ctx)

	lr, err := net.Listen(`tcp`, cfgListenAddr)
	if err != nil {
		return err
	}
	svc.logger.Info().Str(`addr`, cfgListenAddr).Str(`url`, `http://`+cfgListenAddr).Str(`www`, cfgServeRoot).Msg(`starting service`)
	return svr.Serve(lr) // Serve will close the listener.
}

type service struct {
	www          string
	defaultModel string
	websocket    struct {
		upgrader websocket.Upgrader
	}
	// TODO: support for concurrent models.
	// TODO: support for concurrent streams using a single model.
	current struct {
		control sync.Mutex
		model   string
		llm     llm.Model
	}
	logger *zerolog.Logger
}

func (svc *service) handler() http.Handler {
	r := chi.NewRouter()
	r.Use(hog.Middleware())
	if svc.www != `` {
		r.Get(`/*`, svc.handleWWW)
	}
	r.Get(`/v0/models`, handleGet(svc.getModels))
	r.Get(`/v0/predict`, svc.handleGetPredict)
	r.Post(`/v0/predict`, handlePost(svc.handlePredict))
	r.Post(`/v0/encode`, handlePost(svc.handleEncode))
	r.Post(`/v0/decode`, handlePost(svc.handleDecode))
	return r
}

func (svc *service) getModels(ctx context.Context) (status int, rsp struct {
	Error  string               `json:"error,omitempty"`
	Models []modelsResponseItem `json:"models"`
}) {
	dirEntries, err := os.ReadDir(cfgModelRoot)
	if err != nil {
		return 500, rsp
	}
	rsp.Models = make([]modelsResponseItem, 0, len(dirEntries))
	for _, dirEntry := range dirEntries {
		if dirEntry.IsDir() {
			continue
		}
		name := dirEntry.Name()
		if name == `` || name[0] == '.' {
			continue
		}
		if filepath.Ext(name) != `.gguf` {
			continue
		}
		rsp.Models = append(rsp.Models, modelsResponseItem{Model: name})
	}
	return 200, rsp
}

type modelsResponse struct {
	Error  string               `json:"error,omitempty"`
	Models []modelsResponseItem `json:"models"`
}

type modelsResponseItem struct {
	Model string `json:"model"`
}

func (svc *service) handleEncode(ctx context.Context, req *struct {
	Model string `json:"model,omitempty"`
	Text  string `json:"text,omitempty"`
}) (status int, rsp struct {
	Error  string `json:"error,omitempty"`
	Tokens []int  `json:"tokens"`
}) {
	err := svc.withModel(req.Model, func(m llm.Model) error {
		tokens := m.Encode(req.Text)
		rsp.Tokens = make([]int, len(tokens))
		for i, t := range tokens {
			rsp.Tokens[i] = int(t)
		}
		return nil
	})
	if err != nil {
		rsp.Error = err.Error()
		return 500, rsp
	}
	return 200, rsp
}

func (svc *service) handleDecode(ctx context.Context, req *struct {
	Model  string `json:"model,omitempty"`
	Tokens []int  `json:"tokens"`
}) (status int, rsp struct {
	Error string `json:"error,omitempty"`
	Text  string `json:"text"`
}) {
	err := svc.withModel(req.Model, func(m llm.Model) error {
		tokens := make([]llm.Token, len(req.Tokens))
		for i, t := range req.Tokens {
			tokens[i] = llm.Token(t)
		}
		rsp.Text = m.Decode(tokens)
		return nil
	})
	if err != nil {
		rsp.Error = err.Error()
		return 500, rsp
	}
	return 200, rsp
}

func (svc *service) handlePredict(ctx context.Context, req *struct {
	Model    string          `json:"model,omitempty"`
	Settings json.RawMessage `json:"settings,omitempty"`
	Tokens   []int           `json:"tokens,omitempty"` // input tokens, cannot be combined with text
	Text     string          `json:"text,omitempty"`   // input text, cannot be combined with tokens
	Stop     []string        `json:"stop,omitempty"`   // text that stops the prediction
}) (status int, rsp struct {
	Error  string `json:"error,omitempty"`
	Tokens []int  `json:"tokens"`
	Text   string `json:"text"`
}) {
	stopRegex := defaultStopRegex
	if len(req.Stop) > 0 {
		sort.Strings(req.Stop)
		stopRegex = regexp.MustCompile(strings.Join(req.Stop, `|`))
	}
	err := svc.withModel(req.Model, func(m llm.Model) error {
		var tokens []llm.Token
		if len(req.Tokens) > 0 {
			tokens = make([]llm.Token, len(req.Tokens))
			for i, t := range req.Tokens {
				tokens[i] = llm.Token(t)
			}
		} else {
			tokens = m.Encode(req.Text)
		}
		options := make([]predict.Option, 0, 1)
		if len(req.Settings) > 0 {
			options = append(options, predict.JSON(req.Settings))
		}
		stream, err := m.Predict(tokens, options...)
		if err != nil {
			return err
		}
		defer stream.Close()
		for {
			var token llm.Token
			token, err = stream.Next(nil)
			if err == io.EOF {
				return nil
			}
			if err != nil {
				return err
			}
			rsp.Tokens = append(rsp.Tokens, int(token))
			rsp.Text += m.Decode([]llm.Token{token})
			if stopRegex.MatchString(rsp.Text) {
				return nil // TODO: this should be faster.
			}
			hog.From(ctx).Trace().Int(`token`, int(token)).Str(`text`, rsp.Text).Msg(`predicted token`)
		}
	})
	if err != nil {
		rsp.Error = err.Error()
		return 500, rsp
	}
	return 200, rsp
}

func (svc *service) handleGetPredict(w http.ResponseWriter, r *http.Request) {
	conn, err := svc.websocket.upgrader.Upgrade(w, r, nil)
	if err != nil {
		httpErr(w, r, http.StatusBadRequest, err)
		return
	}
	defer conn.Close()
	conn.SetReadLimit(4 << 20) // 4 MiB // TODO: verify that we can widen this after the handshake.
	err = svc.handlePredictConn(r, conn)
	if err != nil {
		hog.For(r).Error().Err(err).Msg(`failed to handle predict request`)
	}
}

func (svc *service) handlePredictConn(r *http.Request, conn *websocket.Conn) error {
	ctx := r.Context()
	var req predictRequest
	err := svc.receiveJSON(ctx, conn, &req)
	if err != nil {
		return err
	}
	err = req.validate(svc)
	if err != nil {
		hog.For(r).Warn().Err(err).Msg(`invalid request`)
		return svc.transmitJSON(ctx, conn, &predictResponse{Error: err.Error()})
	}
	stopRegex, err := compileStopRegex(req.Stop)
	if err != nil {
		hog.For(r).Warn().Err(err).Msg(`invalid stop regex`)
		return svc.transmitJSON(ctx, conn, &predictResponse{Error: err.Error()})
	}
	err = svc.withModel(req.Model, func(m llm.Model) error {
		return svc.predict(ctx, conn, m, &req, stopRegex)
	})
	if err != nil {
		hog.For(r).Warn().Err(err).Msg(``)
		return svc.transmitJSON(ctx, conn, &predictResponse{Error: err.Error()})
	}
	return nil
}

func (svc *service) predict(
	ctx context.Context, conn *websocket.Conn, m llm.Model, req *predictRequest, stopRegex *regexp.Regexp,
) (err error) {
	var tokens []llm.Token
	if len(req.Tokens) > 0 {
		tokens = make([]llm.Token, len(req.Tokens))
		for i, t := range req.Tokens {
			tokens[i] = llm.Token(t)
		}
	} else {
		tokens = m.Encode(req.Text)
	}
	stream, err := m.Predict(tokens)
	if err != nil {
		return err
	}
	defer stream.Close()
	// TODO: defer sending a final event with End=true and Error=err.Error() if err != nil.
	defer func() {
		evt := predictEvent{End: true}
		if err != nil {
			evt.Error = err.Error()
		}
		encErr := svc.transmitJSON(ctx, conn, &evt)
		if err == nil {
			err = encErr
		} else if encErr != nil {
			hog.From(ctx).Warn().Err(encErr).Msg(`failed to send final event`)
		}
	}()
	// TODO: allow the client to interrupt the stream.
	// TODO: allow the client to interlock the stream.  (This would be a guidance-like API.)
	// TODO: support for adjusting temperature and other sampling parameters mid-stream.
	for {
		var token llm.Token
		token, err = stream.Next(nil)
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		evt := predictEvent{Tokens: []int{int(token)}} // TODO: coalesce multiple tokens into a single event.
		evt.Text = m.Decode([]llm.Token{token})
		err = svc.transmitJSON(ctx, conn, &evt)
		if err != nil {
			return err
		}
	}
}

func compileStopRegex(stops []string) (*regexp.Regexp, error) {
	if len(stops) == 0 {
		return defaultStopRegex, nil
	}
	// TODO: option to disallow using regexes for stops.
	for _, stop := range stops {
		_, err := regexp.Compile(stop)
		if err != nil {
			return nil, err
		}
	}
	sort.Strings(stops)
	return regexp.Compile(strings.Join(stops, `|`))
}

var defaultStopRegex = regexp.MustCompile("\n")

func (svc *service) receiveJSON(ctx context.Context, conn *websocket.Conn, v interface{}) error {
	err := conn.SetReadDeadline(time.Now().Add(time.Second * 5)) // TODO: make tunable.
	if err != nil {
		hog.From(ctx).Warn().Err(err).Msg(`failed to impose read deadline`)
	} else {
		defer func() {
			err := conn.SetReadDeadline(time.Time{}) // try to disable deadline
			if err != nil {
				panic(err) // should not fail, since we already set the deadline once.
			}
		}()
	}
	err = conn.ReadJSON(v)
	if err != nil {
		return fmt.Errorf(`failed to decode request: %w`, err)
	}
	hog.From(ctx).Trace().Interface(`msg`, v).Msg(`received JSON`)
	return nil
}

func (svc *service) transmitJSON(ctx context.Context, conn *websocket.Conn, v interface{}) error {
	err := conn.SetWriteDeadline(time.Now().Add(time.Second * 5)) // TODO: make tunable.
	if err != nil {
		hog.From(ctx).Warn().Err(err).Msg(`failed to impose write deadline`)
	} else {
		defer func() {
			err := conn.SetWriteDeadline(time.Time{}) // try to disable deadline
			if err != nil {
				panic(err) // should not fail, since we already set the deadline once.
			}
		}()
	}
	hog.From(ctx).Trace().Interface(`msg`, v).Msg(`transmitting JSON`)
	err = conn.WriteJSON(v)
	if err != nil {
		return fmt.Errorf(`failed to encode response: %w`, err)
	}
	return nil
}

func (svc *service) withModel(model string, fn func(llm.Model) error) error {
	if model != `` {
		// ensure the path does not try to escape the model root
		model = path.Clean(model)
		if len(model) == 0 || model[0] == '.' || model[0] == '/' || model[0] == '\\' {
			return fmt.Errorf(`invalid model name`) // refuse exit from model root
		}
	} else if svc.defaultModel != `` {
		model = svc.defaultModel
	} else {
		return fmt.Errorf(`model not specified`)
	}

	svc.current.control.Lock()
	defer svc.current.control.Unlock()
	if svc.current.model == model {
		return fn(svc.current.llm)
	}
	ext := filepath.Ext(model)
	if ext != `.gguf` {
		return fmt.Errorf(`invalid model name`) // refuse non-gguf models; also stops things like CON / NUL / LPR.
	}
	var err error
	if svc.current.llm != nil {
		svc.current.llm.Close()
	}
	svc.current.model = ``
	svc.current.llm, err = llm.New(filepath.Join(cfgModelRoot, model), llm.Zerolog(*svc.logger))
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			svc.current.llm.Close()
			svc.current.llm = nil
			return
		}
		svc.current.model = model
	}()
	err = fn(svc.current.llm)
	return err
}

func (svc *service) handleWWW(w http.ResponseWriter, r *http.Request) {
	path := filepath.Join(path.Split(strings.TrimPrefix(r.URL.Path, `/`)))
	if path == `` {
		path = `index.html`
	} else if path[0] == '.' { // refuse directory traversal
		http.NotFound(w, r)
		return
	}
	path = filepath.Join(svc.www, path)
	http.ServeFile(w, r, path)
}

type predictRequest struct {
	Model  string   `json:"model,omitempty"`
	Tokens []int    `json:"tokens,omitempty"` // input tokens, cannot be combined with text
	Text   string   `json:"text,omitempty"`   // input text, cannot be combined with tokens
	Stop   []string `json:"stop,omitempty"`   // text that stops the prediction
}

func (req *predictRequest) validate(svc *service) error {
	if req.Model == `` {
		req.Model = svc.defaultModel
	}
	if len(req.Tokens) > 0 && req.Text != `` {
		return fmt.Errorf(`cannot specify both tokens and text`)
	}
	if len(req.Tokens) == 0 && req.Text == `` {
		return fmt.Errorf(`must specify either tokens or text`)
	}
	return nil // TODO
}

type predictResponse struct {
	Error  string `json:"error"`
	Tokens []int  `json:"tokens"`
	Text   string `json:"text"`
}

type predictEvent struct {
	Error  string `json:"error"`
	Tokens []int  `json:"tokens"`
	Text   string `json:"text"`
	End    bool   `json:"end"`
}

func httpErr(w http.ResponseWriter, r *http.Request, code int, err error) {
	httpError(w, r, code, err.Error())
}

func httpError(w http.ResponseWriter, r *http.Request, code int, msg string) {
	log := hog.For(r)
	var evt *zerolog.Event
	if code < 500 {
		evt = log.Warn()
	} else {
		evt = log.Error()
	}
	evt = evt.Int(`code`, code)
	evt.Msg(msg)
	var rsp struct {
		Error string `json:"error"`
	}
	rsp.Error = msg
	js, err := json.Marshal(&rsp)
	if err != nil {
		panic(err)
	}
	writeContent(w, r, code, `application/json`, js)
}

func handleGet[T any](fn func(context.Context) (int, T)) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != `GET` {
			httpErr(w, r, 405, fmt.Errorf(`expected GET`))
			return
		}
		// TODO: verify that they accept JSON
		code, rsp := fn(r.Context())
		writeJSON(w, r, code, &rsp)
	}
}

func handlePost[T, U any](fn func(context.Context, *T) (int, U)) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != `POST` {
			httpErr(w, r, 405, fmt.Errorf(`expected POST`))
			return
		}
		// TODO: verify that they accept JSON
		contentType := strings.SplitN(r.Header.Get(`Content-Type`), `;`, 2)[0]
		if contentType != `application/json` {
			httpErr(w, r, 400, fmt.Errorf(`expected Content-Type: application/json`))
			return
		}
		var req T
		err := json.NewDecoder(r.Body).Decode(&req)
		if err != nil {
			httpErr(w, r, http.StatusBadRequest, err)
			return
		}
		code, rsp := fn(r.Context(), &req)
		writeJSON(w, r, code, &rsp)
	}
}

func writeJSON(w http.ResponseWriter, r *http.Request, code int, v interface{}) {
	js, err := json.Marshal(v)
	if err != nil {
		hog.For(r).Warn().Err(err).Msg(`failed to encode JSON`)
		httpErr(w, r, http.StatusInternalServerError, err)
		return
	}
	writeContent(w, r, code, `application/json`, js)
}

func writeContent(w http.ResponseWriter, r *http.Request, code int, contentType string, content []byte) {
	h := w.Header()
	h.Set(`Content-Type`, contentType)
	h.Set(`Content-Length`, strconv.Itoa(len(content)))
	w.WriteHeader(code)
	_, err := w.Write(content)
	if err != nil {
		hog.For(r).Warn().Err(err).Msg(`failed to send content`)
	}
}
