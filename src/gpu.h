#ifndef NABLA_GPU_H
#define NABLA_GPU_H
#include "gpt.h"
#include <stdio.h>
typedef struct GPUModel GPUModel;
GPUModel *gpu_create(const Gpt *initialized, int batch_capacity, int window_capacity);
/* With an open CSV stream, record synchronized CUDA memory readings during
   initialization. Caller owns the stream. NULL disables profiling. */
GPUModel *gpu_create_profiled(const Gpt *initialized, int batch_capacity,
                              int window_capacity, FILE *memory_csv);
/* Record a named phase (CSV-safe identifier) for a measured model. The delta
   uses the reading with an active CUDA context before cuBLAS initialization. */
int gpu_measure_memory(GPUModel *model, const char *phase, uint64_t step);
int gpu_export(const GPUModel *model, Gpt *dest);
/* Inputs and targets are B x T integers in row-major order. With NULL targets,
   runs inference and returns 0; otherwise returns mean NLL. Errors return NAN.
   Backward accumulates sums across calls, update averages over all tokens. */
double gpu_pass(GPUModel *model, const int *inputs, const int *targets,
                int batch, int length, int backward);
int gpu_last_logits(GPUModel *model, int batch_index, int position, float *host_out);
int gpu_update(GPUModel *model, double lr, double weight_decay, double clip, double *norm);
void gpu_destroy(GPUModel *model);
int gpu_selftest(void);
#endif
