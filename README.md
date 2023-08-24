The `llm-go` package is a Go wrapper for [llama.cpp](https://github.com/ggerganov/llama.cpp) that supports running 
large language models (specifically LLaMA models) in Go.  It was derived from [ollama](https://ollama.ai)'s wrapper
before their shift to embedding [llama-server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)
inside their own server.  (If you need an easy to use local LLaMA server, please, use [ollama.ai](https://ollama.ai).)

*This package is not meant to be API compatible with ollama's (soon to be deprecated) wrapper, nor is its API stable 
yet.  We are still in the middle of a refactor and the GGML to GGUF shift in llama.cpp means work must be done.*

## Quickstart

Assuming you are on a Mac M1 (or M2), have Go and the Apple SDK, and a , the following should "just work":

```shell
wget https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGML/resolve/main/vicuna-7b-v1.5.ggmlv3.q5_K_M.bin
go run github.com/swdunlop/llm-go/examples/io-worker vicuna-7b-v1.5.ggmlv3.q5_K_M.bin
```

From here, you can enter JSON prompts on stdin and get a stream of JSON predictions on stdout.  Sample input:

```javascript
// io-worker consumes JSONL, each object is processed serially and consists of a prompt for prediction.
{"prompt": "What is the softest breed of llama?"}
```

And sample output:

```javascript
// io-worker emits JSONL, strings for incremental predictions, and a final JSON object with timing information
"\n"
" surely"
" not"
" the"
" one"
" that"
" s"
"ells"
" for"
" "
"1"
"."
"5"
" million"
" dollars"
// the final completion has the combined response and wall clock time.
{"response":"\n surely not the one that sells for 1.5 million dollars","seconds":0.942433625}
```

## Supported Platforms

"Support" is a dirty word.  This package is a wrapper around a C++ library that changes very fast.  It works on Mac M1
using Metal acceleration.  We would like it to work on Linux, but we also want to keep the build simple.  (This is why
we based `llm-go` on [ollama](https://ollama.ai)'s wrapper -- they got it working with just Go tools, no makefiles
required.)

On MacOS you will need the Apple SDK for the following frameworks:

- Accelerate
- MetalKit
- MetalPerformanceShaders

If you use Nix on MacOS, our flake should provide all the dependencies you need.  (Keep in mind, if you use Nix to
build, your binary will be linked against the Nix store, which means it will not run on other Macs.)

## Updating GGML or LLaMA

Like the original [ollama](https://ollama.ai) wrapper, `llm-go` currently uses a script to pull in the C++ code and 
headers from a [llama.cpp](https://github.com/ggerganov/llama.cpp) checkout.  This script also prepends Go build tags 
to control which features are built.  (For example, if you don't have Metal acceleration, you can build without it.)
