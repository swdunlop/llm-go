// package slog brokers between log/slog and golang.org/x/exp/slog depending on the Go version.
package slog

import (
	"context"
	"io"
)

func Init(w io.Writer) {
	setDefault(newLogger(newTextHandler(w, &handlerOptions{Level: levelDebug})))
}

func Error(msg string, keyvals ...any) { Background().Error(msg, keyvals...) }
func Warn(msg string, keyvals ...any)  { Background().Warn(msg, keyvals...) }
func Info(msg string, keyvals ...any)  { Background().Info(msg, keyvals...) }
func Debug(msg string, keyvals ...any) { Background().Debug(msg, keyvals...) }

func Background() Interface { return wrap{context.Background(), defaultLogger()} }

func From(ctx context.Context, keyvals ...any) Interface {
	logger, ok := ctx.Value(ctxLogger{}).(Interface)
	if !ok {
		return wrap{ctx, defaultLogger().With(keyvals...)}
	}
	if len(keyvals) == 0 {
		return logger
	}
	return logger.With(keyvals...)
}

func With(ctx context.Context, keyvals ...any) context.Context {
	if len(keyvals) == 0 {
		return ctx
	}
	logger, ok := ctx.Value(ctxLogger{}).(Interface)
	if ok {
		return context.WithValue(ctx, ctxLogger{}, logger.With(keyvals...))
	}
	return context.WithValue(ctx, ctxLogger{}, wrap{ctx, defaultLogger().With(keyvals...)})
}

type ctxLogger struct{}

type wrap struct {
	ctx context.Context
	log *logger
}

func (w wrap) With(keyvals ...any) Interface {
	return wrap{w.ctx, w.log.With(keyvals...)}
}

func (w wrap) Error(msg string, keyvals ...any) { w.log.Log(w.ctx, levelError, msg, keyvals...) }
func (w wrap) Warn(msg string, keyvals ...any)  { w.log.Log(w.ctx, levelWarn, msg, keyvals...) }
func (w wrap) Info(msg string, keyvals ...any)  { w.log.Log(w.ctx, levelInfo, msg, keyvals...) }
func (w wrap) Debug(msg string, keyvals ...any) { w.log.Log(w.ctx, levelDebug, msg, keyvals...) }

type Interface interface {
	With(keyvals ...any) Interface
	Error(msg string, keyvals ...any)
	Warn(msg string, keyvals ...any)
	Info(msg string, keyvals ...any)
	Debug(msg string, keyvals ...any)
}
