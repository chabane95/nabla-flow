#ifndef NABLA_DATA_H
#define NABLA_DATA_H

#include <stddef.h>
#include <stdint.h>

enum { NF_BOS = 256, NF_EOS = 257, NF_VOCAB = 258 };

typedef struct {
    size_t start;
    int length;
} Block;

typedef struct {
    int *tokens;
    Block *blocks;
    size_t records, targets, block_count;
    unsigned char seen[256];
} Corpus;

typedef struct {
    unsigned left, lower, upper;
} Utf8State;

int utf8_byte(Utf8State *state, unsigned byte);
int corpus_load(Corpus *c, const char *path, int context);
/* One weight per entry of tokens. Each record must end in " word.".
   Caller owns the result. Target weights start at block.start + 1. */
int *corpus_answer_weights(const Corpus *c, int factor);
void corpus_free(Corpus *c);
double bigram_nll(const Corpus *train, const Corpus *test);
int data_selftest(void);

#endif
