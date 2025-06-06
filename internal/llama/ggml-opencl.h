// llama.cpp Copyright Georgi Gerganov, 2023, see LICENSE for usage.
// Fetched from https://github.com/ggerganov/llama.cpp/archive/e9c13ff78114af6fc6a4f27cc8dcdda0f3d389fb.tar.gz
// Commit e9c13ff78114af6fc6a4f27cc8dcdda0f3d389fb
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
