package main

import (
	"os"
	"sort"

	"github.com/rs/zerolog"
	zlog "github.com/rs/zerolog/log"
	"github.com/rs/zerolog/pkgerrors"
	"github.com/swdunlop/zugzug-go"
)

func init() {
	zlog.Logger = zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr}).
		With().Timestamp().Logger()
	zerolog.DefaultContextLogger = &zlog.Logger
	zerolog.ErrorStackMarshaler = pkgerrors.MarshalStack

	log := zerolog.New(zerolog.ConsoleWriter{Out: os.Stdout, TimeFormat: `2006-01-02 15:04:05`}).With().Timestamp().Logger()
	zerolog.DefaultContextLogger = &log
	zlog.Logger = log
}

func main() {
	sort.Slice(tasks, func(i, j int) bool { return tasks[i].Name < tasks[j].Name })
	zugzug.Main(tasks)
}

var tasks zugzug.Tasks
