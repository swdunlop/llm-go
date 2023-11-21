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

## Running a Server

The [`cmd/llm`](./cmd/llm) command can also be used to run a server that exposes the `llm-go` package over HTTP.  This
command is largely meant as a proof of concept and will only allow one client to run a prediction at a time.

```shell
$ go run github.com/swdunlop/llm-go/cmd/llm service
```

While working on this package, it is useful to run the server using the `dev` subcommand which runs the `service` as
a child process and restarts it if any Go files change.

```shell
$ go run github.com/swdunlop/llm-go/cmd/llm dev
```

### The `POST /v0/predict` API

A client can `POST` a JSON object to the `/predict` endpoint to request a prediction and the service will block until
the prediction is complete.  The JSON object must have the following structure:

```javascript
{
  // Model identifies the GGUF model to use for prediction, if omitted, the default model will be used if it was
  // specified when the service was started.  (If no default model was specified, the service will return an error.)
  "model": "OpenRP.q5_K_M.gguf",

  // Text provides the text before the prediction, if this is omitted, the service will expect the client to provide
  // a list of tokens using the "tokens" field.
  "text": "Once upon a time.."

  // Tokens provides a list of tokens to use for prediction, as an alternative to providing text.  This cannot be
  // combined with the "text" field.
  "tokens": [7138, 338, 3081],

  // Settings is a list of Model and Prediction settings, see examples/server in the llama.cpp project for 
  // documentation.
  "settings": {
    "temperature": 0.0
  }

  // Stop provides a list of regular expressions that, when found, will cause the prediction to stop.  This defaults to
  // "\n" (a newline) if omitted.  The supported regular expression syntax is that of the Go standard library.
  "stop": ["\n"],
}
```

The `settings` field provides the same options as [examples/server]() from the [llama.cpp]() project, with the exception of options like `stream` that affects how the prediction is returned.

[examples/server]: https://github.com/ggerganov/llama.cpp/tree/master/examples/server#api-endpoints
[llama.cpp]: https://github.com/ggerganov/llama.cpp

The response will have the following structure:

```javascript
{
    // Error will be set if there was an error during prediction to an English string describing the error.  If this
    // is set, other fields may be omitted.
    "error": "",

    // Text is the text that was generated by the model.
    "text": "The more you know, the better off you are.",

    // Tokens is a list of tokens that were generated by the model.
    "tokens": [7138, 338, 3081],
}
```


### The `GET /v0/predict` API

A client can also use `GET` to establish a WebSocket connection to the `/predict` endpoint, send a request and receive
a prediction stream.  This is useful that want to stream the prediction as it is generated.  The first request must
have the same structure as the `POST /v0/predict` request described above.

Responses from the prediction API has the same structure as the `POST /v0/predict` response described above, with the
addition of a boolean field, `end`, that will be `true` when the prediction is complete with a complete prediction
including all of the text and generated tokens.

<!-- TODO: This is likely a bad idea, it would be better to include Guidance templates or support for BNF grammars, or
both.

### The `GET /v0/guide` API

The `GET /v0/guide` endpoint is similar to the `GET /v0/predict` endpoint, but it lets the client "guide" the prediction
after each batch of generated tokens.  This is a very advanced feature that lets advanced clients rewind the prediction
context or insert additional text or tokens after a batch of predicted tokens.

This API is inherently more complex than the `/v0/predict` APIs and has increased latency since the service must wait
for the client to respond before it can continue the prediction.
-->

### The `GET /v0/models` API

The `GET /v0/models` endpoint returns a list of the GGUF models that are available to the service from the model
directory and will identify the default model.

```javascript
{
  // Default identifies the default model, if any.
  "default": "OpenRP.q5_K_M.gguf",

  // Models is a list of the models that are available to the service.
  "models": [
    {"model": "OpenRP.q5_K_M.gguf"}
  ]
}
```

### The `POST /v0/encode` API

The `POST /v0/encode` API encodes a string of text into a list of tokens.  This is useful for clients that operate on
tokens directly.  The request has the following structure:

```javascript
{
  // Model identifies the GGUF model to use for prediction, if omitted, the default model will be used if it was
  // specified when the service was started.  (If no default model was specified, the service will return an error.)
  "model": "OpenRP.q5_K_M.gguf",

  // Text provides the text to encode.
  "text": "Once upon a time.."
}
```

The response will have the following structure:

```javascript
{
    // Error will be set if there was an error during prediction to an English string describing the error.  If this
    // is set, other fields may be omitted.
    "error": "",

    // Tokens is a list of tokens that were generated by the model.
    "tokens": [9038, 2501, 263, 931, 636],
}
```

### The `POST /v0/decode` API

The `POST /v0/decode` API decodes a list of tokens into a string of text.  Like the `POST /v0/encode` API, this is
meant for clients that operate on tokens directly.  The request has the following structure:

```javascript
{
  // Model identifies the GGUF model to use for prediction, if omitted, the default model will be used if it was
  // specified when the service was started.  (If no default model was specified, the service will return an error.)
  "model": "OpenRP.q5_K_M.gguf",

  // Tokens is a list of tokens to decode.
  "tokens": [9038, 2501, 263, 931, 636]
}
```

The response will have the following structure:

```javascript
{
    // Error will be set if there was an error during prediction to an English string describing the error.  If this
    // is set, other fields may be omitted.
    "error": "",

    // Text is the text that was generated by the model.
    "text": " Once upon a time..",
    // Note that this isn't exactly the same text as the previous example -- LLaMA's tokens often include a leading
    // space.
}
```


## Updating Llama.cpp

This package embeds a copy of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) (which in turns embeds GGML) in 
[`internal/llama`](./internal/llama).  To update this copy, run the `bump` subcommand with a commitish from the
upstream repository.

```shell
$ go run github.com/swdunlop/llm-go/cmd/llm bump -c bc34dd4f5b5a7c10ae3ed85a265ce6f2ed2fab79
```

Frequently, this will cause issues with a patch in this repository for MacOS that embeds the content of 
`ggml-metal.metal` into the package so it can be loaded from memory.  (The upstream [llama.cpp]() project expects you
to provide this file at runtime.)  If this happens, you can run the `bump` subcommand with the `--no-patch` flag to 
not update the `ggml-metal.m` file and then recreate the patch.
