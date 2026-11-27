#include "data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reject surrogates, overlong encodings and scalars beyond U+10FFFF. */
int utf8_byte(Utf8State *s, unsigned b)
{
    if (s->left) {
        if (b < s->lower || b > s->upper)
            return 0;
        --s->left;
        s->lower = 0x80;
        s->upper = 0xbf;
        return 1;
    }
    if (b <= 0x7f)
        return 1;
    if (b < 0xc2 || b > 0xf4)
        return 0;
    s->left = b < 0xe0 ? 1u : (b < 0xf0 ? 2u : 3u);
    s->lower = b == 0xe0 ? 0xa0u : (b == 0xf0 ? 0x90u : 0x80u);
    s->upper = b == 0xed ? 0x9fu : (b == 0xf4 ? 0x8fu : 0xbfu);
    return 1;
}

void corpus_free(Corpus *c)
{
    free(c->tokens);
    free(c->blocks);
    *c = (Corpus){0};
}

/* A block always has length+1 neighboring tokens: its inputs start at
   start, its targets at start+1. Each byte and EOS is predicted once. */
static int from_bytes(Corpus *c, const unsigned char *bytes, size_t size, int context)
{
    if (size > SIZE_MAX / (3 * sizeof(int)) || context < 1)
        return 0;
    size_t begin = size >= 3 && memcmp(bytes, "\xef\xbb\xbf", 3) == 0 ? 3 : 0;
    for (int pass = 0; pass < 2; ++pass) {
        size_t used = 0, block = 0;
        for (size_t i = begin; i < size;) {
            size_t first = i;
            while (i < size && bytes[i] != '\n' && bytes[i] != '\r')
                ++i;
            size_t length = i - first;
            while (i < size && (bytes[i] == '\n' || bytes[i] == '\r'))
                ++i;
            if (!length)
                continue;
            if (pass == 0) {
                Utf8State state = {0};
                for (size_t j = first; j < first + length; ++j)
                    if (!bytes[j] || !utf8_byte(&state, bytes[j]))
                        return 0;
                if (state.left)
                    return 0;
                ++c->records;
                c->targets += length + 1;
                c->block_count += length / (size_t)context + 1;
            } else {
                size_t start = used, left = length + 1;
                c->tokens[used++] = NF_BOS;
                for (size_t j = first; j < first + length; ++j) {
                    c->tokens[used++] = bytes[j];
                    c->seen[bytes[j]] = 1;
                }
                c->tokens[used++] = NF_EOS;
                while (left) {
                    size_t n = left > (size_t)context ? (size_t)context : left;
                    c->blocks[block++] = (Block){start, (int)n};
                    start += n;
                    left -= n;
                }
            }
        }
        if (pass == 0) {
            if (!c->records || c->block_count > SIZE_MAX / sizeof(Block))
                return 0;
            c->tokens = malloc((c->targets + c->records) * sizeof(*c->tokens));
            c->blocks = malloc(c->block_count * sizeof(*c->blocks));
            if (!c->tokens || !c->blocks)
                return 0;
        }
    }
    return 1;
}

int corpus_load(Corpus *c, const char *path, int context)
{
    FILE *f = fopen(path, "rb");
    long end = -1;
    if (f && fseek(f, 0, SEEK_END) == 0)
        end = ftell(f);
    unsigned char *raw = end > 0 ? malloc((size_t)end) : NULL;
    int ok = raw && fseek(f, 0, SEEK_SET) == 0 && fread(raw, 1, (size_t)end, f) == (size_t)end &&
             from_bytes(c, raw, (size_t)end, context);
    free(raw);
    if (f && fclose(f) != 0)
        ok = 0;
    if (!ok) {
        fprintf(stderr, "Cannot read nonempty UTF-8 records from %s\n", path);
        corpus_free(c);
    }
    return ok;
}

int *corpus_answer_weights(const Corpus *c, int factor)
{
    size_t count = c->targets + c->records;
    if (factor < 1) return NULL;
    int *weights = malloc(count * sizeof(*weights));
    if (!weights) return NULL;
    for (size_t i = 0; i < count; ++i) weights[i] = 1;
    size_t record = 0;
    for (size_t eos = 0; eos < count; ++eos) {
        if (c->tokens[eos] != NF_EOS) continue;
        if (eos < record + 4 || c->tokens[eos-1] != '.') goto invalid;
        size_t end = eos-1, start = end;
        while (start > record+1 && c->tokens[start-1] != ' ' && c->tokens[start-1] != '\t')
            --start;
        if (start == end || start == record+1) goto invalid;
        for (size_t i = start; i < end; ++i) weights[i] = factor;
        record = eos + 1;
    }
    return weights;
invalid:
    fprintf(stderr,"Missing final space-delimited answer word and period in training record.\n");
    free(weights);
    return NULL;
}

double bigram_nll(const Corpus *train, const Corpus *test)
{
    /* Same 258-symbol output alphabet as the GPT; add-0.1 smoothing. */
    size_t *counts = calloc(NF_VOCAB * NF_VOCAB, sizeof(*counts));
    size_t totals[NF_VOCAB] = {0};
    if (!counts)
        return NAN;
    for (size_t n = 0; n < train->block_count; ++n) {
        Block b = train->blocks[n];
        const int *t = train->tokens + b.start;
        for (int i = 0; i < b.length; ++i) {
            ++counts[t[i] * NF_VOCAB + t[i+1]];
            ++totals[t[i]];
        }
    }
    double loss = 0.0;
    for (size_t n = 0; n < test->block_count; ++n) {
        Block b = test->blocks[n];
        const int *t = test->tokens + b.start;
        for (int i = 0; i < b.length; ++i) {
            size_t count = counts[t[i] * NF_VOCAB + t[i+1]];
            loss -= log(((double)count + 0.1) / ((double)totals[t[i]] + 0.1 * NF_VOCAB));
        }
    }
    free(counts);
    return loss / (double)test->targets;
}

static int test_answer_weights(void)
{
    static const unsigned char text[] = "X A\xc3\xa4.\r\nY B.";
    static const int expected[] = {1,1,1,4,4,4,1,1,1,1,1,4,1,1};
    Corpus c = {0};
    int ok = from_bytes(&c,text,sizeof(text)-1,2);
    int *weights = ok ? corpus_answer_weights(&c,4) : NULL;
    ok = weights && c.targets+c.records == sizeof(expected)/sizeof(*expected) &&
         memcmp(weights,expected,sizeof(expected)) == 0;
    free(weights);
    corpus_free(&c);
    return ok;
}

int data_selftest(void)
{
    static const unsigned char raw[] = "\xef\xbb\xbf" "Ab\r\n\n" "\xc3\xa4" "x";
    static const unsigned char bad[] = {0xed, 0xa0, 0x80};
    static const int tokens[] = {NF_BOS,65,98,NF_EOS,NF_BOS,195,164,120,NF_EOS};
    Corpus c = {0}, invalid = {0};
    int ok = test_answer_weights() && from_bytes(&c, raw, sizeof(raw)-1, 2) && c.records == 2 && c.targets == 7 &&
             c.block_count == 4 && memcmp(c.tokens, tokens, sizeof(tokens)) == 0 &&
             c.blocks[0].start == 0 && c.blocks[0].length == 2 &&
             c.blocks[1].start == 2 && c.blocks[1].length == 1 &&
             c.blocks[2].start == 4 && c.blocks[2].length == 2 &&
             c.blocks[3].start == 6 && c.blocks[3].length == 2 &&
             !from_bytes(&invalid, bad, sizeof(bad), 4);
    if (ok) {
        double expected = (2 * log((2 + 0.1*NF_VOCAB)/1.1) +
                           5 * log((1 + 0.1*NF_VOCAB)/1.1)) / 7;
        ok = fabs(bigram_nll(&c, &c) - expected) < 1e-12;
    }
    corpus_free(&invalid);
    corpus_free(&c);
    printf("UTF-8, boundaries, block targets, answer weights, bigram likelihood: %s\n", ok ? "ok" : "FAILED");
    return ok;
}
