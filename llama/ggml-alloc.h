/**
 * llama.cpp - git 744f4916b35f5a12f200e7fd4bc6a3a3ecf43c0d
 *
 * MIT License
 *
 * Copyright (c) 2023 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif


GGML_API struct ggml_allocr * ggml_allocr_new(void * data, size_t size, size_t alignment);
GGML_API struct ggml_allocr * ggml_allocr_new_measure(size_t alignment);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
GGML_API void   ggml_allocr_set_parse_seq(struct ggml_allocr * alloc, const int * list, int n);

GGML_API void   ggml_allocr_free(struct ggml_allocr * alloc);
GGML_API bool   ggml_allocr_is_measure(struct ggml_allocr * alloc);
GGML_API void   ggml_allocr_reset(struct ggml_allocr * alloc);
GGML_API void   ggml_allocr_alloc(struct ggml_allocr * alloc, struct ggml_tensor * tensor);
GGML_API size_t ggml_allocr_alloc_graph(struct ggml_allocr * alloc, struct ggml_cgraph * graph);


#ifdef  __cplusplus
}
#endif
