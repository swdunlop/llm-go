// llama.cpp Copyright Georgi Gerganov, 2023, see LICENSE for usage.
// Fetched from https://github.com/ggerganov/llama.cpp/archive/11bff290458f12f020b588792707f76ec658a27a.tar.gz
// Commit 11bff290458f12f020b588792707f76ec658a27a
//go:build opencl
#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_cl_init(void);

void   ggml_cl_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

void * ggml_cl_host_malloc(size_t size);
void   ggml_cl_host_free(void * ptr);

void ggml_cl_free_data(const struct ggml_tensor* tensor);

void ggml_cl_transform_tensor(void * data, struct ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
