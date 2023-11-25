// llama.cpp Copyright Georgi Gerganov, 2023, see LICENSE for usage.
// Fetched from https://github.com/ggerganov/llama.cpp/archive/e9c13ff78114af6fc6a4f27cc8dcdda0f3d389fb.tar.gz
// Commit e9c13ff78114af6fc6a4f27cc8dcdda0f3d389fb
//go:build mpi
#pragma once

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_mpi_context;

void ggml_mpi_backend_init(void);
void ggml_mpi_backend_free(void);

struct ggml_mpi_context * ggml_mpi_init(void);
void ggml_mpi_free(struct ggml_mpi_context * ctx);

int ggml_mpi_rank(struct ggml_mpi_context * ctx);

void ggml_mpi_eval_init(
        struct ggml_mpi_context * ctx_mpi,
                            int * n_tokens,
                            int * n_past,
                            int * n_threads);

void ggml_mpi_graph_compute_pre(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers);

void ggml_mpi_graph_compute_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers);

#ifdef __cplusplus
}
#endif
