The `llm-go` package is a Go wrapper for [`llama.cpp`](https://github.com/ggerganov/llama.cpp) derived from an early
version of [ollama](https://ollama.ai) that is intended to be easier to maintain and use in Go projects.

## License

This package is licensed under the MIT license.  See [`LICENSE`](./LICENSE) for more information.

## Command Line Usage

This package comes with an command line utility, [`cmd/llm`](./cmd/llm), that can demonstrates how to use the `llm-go`
package to predict text.  (This utility also includes a number of other subcommands for development, such as the `bump`
subcommand explained below, see [`cmd/llm/dev.go`](./cmd/llm/dev.go) for more information.)

```shell
$ go run github.com/swdunlop/llm-go/cmd/llm predict -m OpenRP.q5_K_M.gguf Once upon a time.. 2>/dev/null
```
```
The story of the little girl who was born with a silver spoon in her mouth is well known. But what about the little boy who was born with a golden one? This is his story.

In a small village, there lived a poor family. The father worked as a blacksmith and the mother took care of their only son, named Jack. They were happy despite their poverty because they had each other's love and support.

One day, while working at his forge, the blacksmith found an old chest buried under some rubble. Inside it was a golden spoon. He knew that this discovery would change their lives forever.

The next morning, he woke up early to tell his wife about the treasure they had found. She couldn't believe her eyes when she saw the shiny object. They decided to keep it hidden from everyone else until they figured out how to use it wisely.

...
```

## Go Usage

The [llm-go](https://pkg.go.dev/github.com/swdunlop/llm-go) package is relatively simple to use:

```go
// Like llama.cpp, the llm-go package loads its model from GGUF format files.
m, _ := llm.Load("OpenRP.q5_K_M.gguf")
defer m.Close() // Ensures we release the memory associated with the model.
input := m.Encode("Once upon a time..")
s, _ := m.Predict(input)
defer s.Close() // Ensures we releaswe the memory associated with the prediction context.
var output []llm.Token
for {
    // Next produces one predicted token at a time.  You can add more tokens to the context if you want, this is 
    // helpful for implementing things like Guidance.
    next, err := m.Next(nil)
    if err != nil {
        break
    }

    // Each time Next is returned, it will return a single token or an error.  The error is usually either io.EOF
    // when the model has reached the end of its prediction (by returning the end of stream (EOS) token) or 
    // llm.Overflow when the context is full and the model cannot predict any more tokens.
    output = append(output, next)
}
fmt.Println(m.Decode(output))
```

## Updating Llama.cpp

This package embeds a copy of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) (which in turns embeds GGML) in 
[`internal/llama`](./internal/llama).  To update this copy, run the `bump` subcommand with a commitish from the
upstream repository.

```shell
$ go run github.com/swdunlop/llm-go/cmd/llm bump -c bc34dd4f5b5a7c10ae3ed85a265ce6f2ed2fab79
```
