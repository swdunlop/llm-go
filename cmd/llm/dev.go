//go:build !release
// +build !release

package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/swdunlop/zugzug-go"
	"github.com/swdunlop/zugzug-go/zug"
	"github.com/swdunlop/zugzug-go/zug/console"
	"github.com/swdunlop/zugzug-go/zug/parser"
)

// This file contains development tasks that are not useful in a released version
// of nats-llama.go and are therefore excluded from the release build.

func init() {
	zug.Debug() // let zug tasks panic and print stack traces

	tasks = append(tasks, zugzug.Tasks{
		{Name: "bump", Fn: bumpLlama, Use: "bumps llama.cpp to the specified commitish", Parser: parser.New(
			parser.String(&bumpCommitish, "commitish", "c", "commitish to bump to"),
			parser.Bool(&noPatch, "no-patch", "", "do not patch the llama.cpp source files aside from adding metadata"),
		)},
		{Name: "dev", Fn: runDev, Use: "runs the service locally, restarting on update", Parser: parser.Custom()},
	}...)
}

var (
	bumpCommitish string
	noPatch       bool
)

func bumpLlama(ctx context.Context) (err error) {
	println(`.. bumping llama.cpp to`, bumpCommitish)
	err = cleanLlama(ctx)
	if err != nil {
		return err
	}
	url := fmt.Sprintf("https://github.com/ggerganov/llama.cpp/archive/%s.tar.gz", bumpCommitish)
	req, err := http.NewRequestWithContext(ctx, `GET`, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf(`%v while fetching %v`, resp.Status, url)
	}
	gr, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf(`%w while decompressing %v`, err, url)
	}
	defer gr.Close()
	tr := tar.NewReader(gr)

	cLicense := fmt.Sprintf("// llama.cpp Copyright %v, %v, see LICENSE for usage.\n// Fetched from %v\n// Commit %v\n",
		`Georgi Gerganov`,
		time.Now().Format(`2006`),
		url,
		bumpCommitish,
	)

	dir := filepath.Join(`internal`, `llama`)
	err = os.MkdirAll(dir, 0755)
	if err != nil {
		return err
	}
	licensed := false
	defer func() {
		if licensed {
			return
		}
		_ = cleanLlama(ctx)
	}()
	for {
		hdr, err := tr.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf(`%w while reading %v`, err, url)
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		// trim off "llama.cpp-$commitish" nonsense
		m := strings.SplitN(hdr.Name, `/`, 2)
		if len(m) < 2 {
			continue
		}
		hdr.Name = m[1]

		name := path.Base(hdr.Name)
		if name != hdr.Name {
			continue
		}
		if name == `LICENSE` {
			h := sha256.New()
			_, _ = io.Copy(h, tr)
			hash := hex.EncodeToString(h.Sum(nil))
			if hash != mitLicenseHash {
				return fmt.Errorf(`unexpected license hash %v`, hash)
			}
			licensed = true
			continue
		}
		if name == "build-info.h" {
			continue
		}

		ext := path.Ext(name)
		license, tag := ``, ``
		switch strings.ToLower(ext) {
		case `.c`, `.h`, `.cpp`, `.m`, `.metal`, `.cu`:
			license = cLicense
			switch name[:len(name)-len(ext)] {
			case `ggml-metal`:
				tag = "//go:build darwin\n"
			case `ggml-mpi`:
				tag = "//go:build mpi\n" // untested
			case `ggml-opencl`:
				tag = "//go:build opencl\n" // untested
			}
		}
		// TODO: copy LICENSE file

		if license == `` {
			continue
		}
		var buf bytes.Buffer
		buf.Grow(int(hdr.Size) + 1024)
		buf.WriteString(license)
		if tag != `` {
			buf.WriteString(tag)
		}
		_, err = io.Copy(&buf, tr)
		if err != nil {
			return fmt.Errorf(`%w while reading %q from %v`, err, hdr.Name, url)
		}
		if err := os.WriteFile(filepath.Join(dir, name), buf.Bytes(), 0644); err != nil {
			return err
		}
	}
	err = generateLlamaMetalC(ctx)
	if err != nil {
		return err
	}
	err = patchGgmlMetal(ctx)
	if err != nil {
		return err
	}
	return nil
}

const mitLicenseHash = `e562a2ddfaf8280537795ac5ecd34e3012b6582a147ef69ba6a6a5c08c84757d`

func cleanLlama(ctx context.Context) error {
	dir := filepath.Join(`internal`, `llama`)
	if _, err := os.Stat(dir); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		return nil
	}
	matches, err := filepath.Glob(filepath.Join(dir, `*.go`))
	if err != nil {
		return err
	}
	more, err := filepath.Glob(filepath.Join(dir, `*.md`))
	if err != nil {
		return err
	}
	matches = append(matches, more...)
	more, err = filepath.Glob(filepath.Join(dir, `*.patch`))
	if err != nil {
		return err
	}
	matches = append(matches, more...)
	matches = append(matches, filepath.Join(dir, `LICENSE`))
	protect := make(map[string]struct{}, len(matches))
	for _, m := range matches {
		protect[m] = struct{}{}
	}
	return filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if path == dir {
			return nil
		}
		if path[0] == '.' {
			return nil
		}
		if _, ok := protect[path]; ok {
			return nil
		}
		return os.Remove(path)
	})
}

func generateLlamaMetalC(ctx context.Context) error {
	data, err := os.ReadFile(filepath.Join(`internal`, `llama`, `ggml-metal.metal`))
	if err != nil {
		return err
	}
	lines := strings.Split(string(data), "\n")
	out := make([]string, 0, len(lines)+8)
	out = append(out, "// Code generated by nats-llama; DO NOT EDIT.")
	out = append(out, "//go:build darwin")
	out = append(out, "// +build darwin")
	out = append(out, "#import <Foundation/Foundation.h>")
	out = append(out, "NSString * const ggml_metal_src = @")
	replacer := strings.NewReplacer(
		`\`, `\\`,
		`"`, `\"`,
		"\uFEFF", ``, // Death to the BOM
	)
	for _, line := range lines {
		out = append(out, "\t\""+replacer.Replace(line)+`\n"`)
	}
	out = append(out, ";")
	return os.WriteFile(filepath.Join(`internal`, `llama`, `ggml-metal-metal.m`), []byte(strings.Join(out, "\n")), 0644)
}

func patchGgmlMetal(ctx context.Context) error {
	if noPatch {
		println(`.. skipping patches`)
		return nil
	}
	err := console.Run(ctx, `patch`, filepath.Join(`internal`, `llama`, `ggml-metal.m`), filepath.Join(`internal`, `llama`, `ggml-metal.patch`))
	if err != nil {
		return err
	}
	return nil
}

func runDev(ctx context.Context) error {
	args := []string{
		`--regex=\.go$`,
		`--start-service`,
		`--decoration=none`,
		`--`,
		`go`, `run`, `./cmd/llm`, `service`, `-www`, `./www`,
	}
	args = append(args, parser.Args(ctx)...)
	return goRun(ctx, `github.com/cespare/reflex`, args...)
}

func goRun(ctx context.Context, pkg string, args ...string) error {
	return console.Run(ctx, `go`, append([]string{`run`, pkg}, args...)...)
}
