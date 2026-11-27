#ifndef NABLA_FLOW_GPT_H
#define NABLA_FLOW_GPT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int vocab_size, context, width, heads, layers;
} GptConfig;

/* Initialize with {0}, release with gpt_free. Parameters are contiguous floats
   in deterministic order and can be serialized with the configuration. */
typedef struct {
    GptConfig config;
    size_t parameter_count;
    float *parameters;
    uint64_t steps;
    struct GptState *state;
} Gpt;

size_t gpt_parameter_count(GptConfig config);
int gpt_init(Gpt *model, GptConfig config, uint32_t seed);
void gpt_free(Gpt *model);
int gpt_forward(Gpt *model, const int *tokens, int length);
const float *gpt_logits(const Gpt *model, int position);
/* Copies length probabilities after forward, with exact zeros beyond position. */
int gpt_attention_row(const Gpt *model, int layer, int head, int position, float *out);
/* Targets are shifted by the caller. Each function returns mean token loss;
   backward also accumulates the SUM of the token gradients for this sequence. */
double gpt_loss(Gpt *model, const int *targets);
double gpt_backward(Gpt *model, const int *targets);
/* Positive integer weight per target (NULL means all ones). Returns weighted
   mean loss. AdamW normalizes accumulated gradient sums by total weight. */
double gpt_backward_weighted(Gpt *model, const int *targets, const int *weights);
/* Raw accumulated gradient sums in parameter order; NULL before first backward. */
const float *gpt_gradient(const Gpt *model);
void gpt_zero_grad(Gpt *model);
/* Divide by accumulated target weight, clip global L2 norm, then AdamW.
   clip_norm == 0 disables clipping. grad_norm optionally receives the raw
   norm of the mean gradient. A successful step clears accumulated gradients. */
int gpt_step(Gpt *model, double lr, double weight_decay, double clip_norm, double *grad_norm);
int gpt_selftest(void);

#endif
