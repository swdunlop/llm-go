//go:build !go1.21
// +build !go1.21

// Package slog uses either the experimental slog package or the standard slog package depending on the Go version.
// (slog indeed, this package took a really long time to reach the standard library.)
package slog

import (
	"io"

	"golang.org/x/exp/slog"
)

type logger = slog.Logger
type handlerOptions = slog.HandlerOptions

func defaultLogger() *logger           { return slog.Default() }
func newLogger(h slog.Handler) *logger { return slog.New(h) }
func setDefault(l *logger)             { slog.SetDefault(l) }
func newTextHandler(w io.Writer, opts *handlerOptions) slog.Handler {
	return slog.NewTextHandler(w, opts)
}

const (
	levelDebug = slog.LevelDebug
	levelInfo  = slog.LevelInfo
	levelWarn  = slog.LevelWarn
	levelError = slog.LevelError
)
