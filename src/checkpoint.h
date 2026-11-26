#ifndef NABLA_CHECKPOINT_H
#define NABLA_CHECKPOINT_H

#include "gpt.h"

/* Inference weights. The byte-alphabet bitmap is optional for token-ID models. */
int checkpoint_save(const char *path, const Gpt *g, const unsigned char seen[256]);
int checkpoint_load(const char *path, Gpt *g, unsigned char seen[256]);
int checkpoint_selftest(void);

#endif
