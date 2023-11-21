#!/usr/bin/env hurl --test --variable base=http://localhost:7272

# These tests require https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q3_K_M.gguf
# to be downloaded and placed in the models directory.  This model was not selected for its utility, just
# for its size and being commonly available.

# Get a list of models
GET {{base}}/v0/models

HTTP 200
Content-Type: application/json
[Asserts]
jsonpath "$.error" not exists
jsonpath "$.models[*].model" includes "llama-2-7b.Q3_K_M.gguf"

# Encode a message with the model
POST {{base}}/v0/encode
Content-Type: application/json
{
    "model": "llama-2-7b.Q3_K_M.gguf",
    "text": "Hello, world!"
}

HTTP 200
Content-Type: application/json
[Asserts]
jsonpath "$.error" not exists
jsonpath "$.tokens" count == 4
jsonpath "$.tokens[0]" == 15043
jsonpath "$.tokens[1]" == 29892
jsonpath "$.tokens[2]" == 3186
jsonpath "$.tokens[3]" == 29991

# Decode a message with the model
POST {{base}}/v0/decode
Content-Type: application/json
{
    "model": "llama-2-7b.Q3_K_M.gguf",
    "tokens": [15043, 29892, 3186, 29991]
}

HTTP 200
Content-Type: application/json
[Asserts]
jsonpath "$.error" not exists
jsonpath "$.text" == " Hello, world!" # Note the leading space

# Predict a response without using streaming.
POST {{base}}/v0/predict
Content-Type: application/json
{
    "model": "llama-2-7b.Q3_K_M.gguf",
    "text": "What is the cube root of 27?\n",
    "options": {
        "temperature": 0
    },
    "stop": ["\n"]
}

HTTP 200
Content-Type: application/json
[Asserts]
jsonpath "$.error" not exists
jsonpath "$.text" == " The cube root of 27 is 3.\n"

# Predict the response again.  This should be faster since the input is cached, so only the output is generated.
POST {{base}}/v0/predict
Content-Type: application/json
{
    "model": "llama-2-7b.Q3_K_M.gguf",
    "text": "What is the cube root of 27?\n",
    "options": {
        "temperature": 0
    },
    "stop": ["\n"]
}

HTTP 200
Content-Type: application/json
[Asserts]
jsonpath "$.error" not exists
jsonpath "$.text" == " The cube root of 27 is 3.\n"

# This time, continue on from the previous response, tweaking the intervening whitespace.
# Llama will start predicting more cube roots (since it was not chat trained or given instructions).
POST {{base}}/v0/predict
Content-Type: application/json
{
    "model": "llama-2-7b.Q3_K_M.gguf",
    "text": "What is the cube root of 27?\nThe cube root of 27 is 3.\n",
    "options": {
        "temperature": 0
    },
    "stop": ["\n"]
}

HTTP 200
Content-Type: application/json
[Asserts]
jsonpath "$.error" not exists
jsonpath "$.text" == " The cube root of 27 is 3.\n"
