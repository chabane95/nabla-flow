#ifndef NABLA_CHECKPOINT_H
#define NABLA_CHECKPOINT_H

#include "gpt.h"

/* Inference weights. The byte-alphabet bitmap is optional for token-ID models. */
int checkpoint_save(const char *path, const Gpt *g, const unsigned char seen[256]);
int checkpoint_load(const char *path, Gpt *g, unsigned char seen[256]);
/* FNV-1a of parameter bytes alone, in little-endian checkpoint order. */
uint64_t checkpoint_parameter_hash(const Gpt *g);
int checkpoint_selftest(void);

#endif
