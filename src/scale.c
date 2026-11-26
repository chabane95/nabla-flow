/* Full GPT-2 Small dimensions. The input format is a headerless stream of
   GPT-2 BPE token IDs (uint16 little-endian), as opposed to the byte demo. */
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#endif
#include "gpt.h"
#include "checkpoint.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#define seek64 _fseeki64
#define tell64 _ftelli64
#else
#include <sys/types.h>
#define seek64 fseeko
#define tell64 ftello
#endif

static const GptConfig small = {50257, 1024, 768, 12, 12};
static const double rate = 0.0003;

static int number(const char *s, int lo, int hi, int *out)
{
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || end == s || *end || n < lo || n > hi) return 0;
    *out = (int)n;
    return 1;
}

typedef struct { FILE *file; int64_t count; } Shard;

static int open_shard(Shard *s, const char *path)
{
    s->file = fopen(path, "rb");
    if (!s->file || seek64(s->file, 0, SEEK_END)) return 0;
    int64_t bytes = tell64(s->file);
    if (bytes < 4 || bytes % 2) return 0;
    s->count = bytes / 2;
    return 1;
}

static int read_window(Shard *s, int64_t start, int n, int *ids)
{
    unsigned char bytes[2 * (1024 + 1)];
    if (start < 0 || n < 1 || n > 1025 || start + n > s->count ||
        seek64(s->file, 2 * start, SEEK_SET) ||
        fread(bytes, 2, (size_t)n, s->file) != (size_t)n) return 0;
    for (int i = 0; i < n; ++i) {
        ids[i] = bytes[2*i] | ((int)bytes[2*i+1] << 8);
        if (ids[i] >= small.vocab_size) return 0;
    }
    return 1;
}

static double heldout(Gpt *g, Shard *s, int window)
{
    int ids[1025];
    double sum = 0.0;
    for (int64_t start = 0; start < s->count - 1; ) {
        int64_t left = s->count - 1 - start;
        int n = left < window ? (int)left : window;
        if (!read_window(s, start, n + 1, ids) || !gpt_forward(g, ids, n)) return NAN;
        sum += n * gpt_loss(g, ids + 1);
        start += n;
    }
    return sum / (double)(s->count - 1);
}

static uint64_t random64(uint64_t *state)
{
    uint64_t x = (*state += UINT64_C(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static int train_scale(const char *tr_path, const char *va_path, int steps, int window,
                       const char *model_path, int batch)
{
    Gpt g = {0};
    Shard tr = {0}, va = {0};
    int ids[1025], ok = 0;
    uint64_t rng = 42;
    if (!strcmp(tr_path, va_path) || !strcmp(tr_path, model_path) || !strcmp(va_path, model_path))
        goto done;
    /* Create a new output only. This also prevents a checkpoint aliasing a shard. */
    FILE *probe = fopen(model_path, "rb");
    if (probe) { fclose(probe); fputs("Choose a new checkpoint path.\n", stderr); goto done; }
    if (!open_shard(&tr, tr_path) || !open_shard(&va, va_path) || tr.count <= window ||
        !gpt_init(&g, small, 42)) goto done;
    double best = heldout(&g, &va, window);
    if (!isfinite(best) || !checkpoint_save(model_path, &g, NULL)) goto done;
    printf("step,val_nll,best_val_nll\n0,%.9g,%.9g\n", best, best);
    for (int step = 1; step <= steps; ++step) {
        for (int b = 0; b < batch; ++b) {
            int64_t at = (int64_t)(random64(&rng) % (uint64_t)(tr.count - window));
            if (!read_window(&tr, at, window + 1, ids) || !gpt_forward(&g, ids, window) ||
                !isfinite(gpt_backward(&g, ids + 1))) goto done;
        }
        if (!gpt_step(&g, rate, 0.1, 1.0, NULL)) goto done;
        if (step % 100 == 0 || step == steps) {
            double val = heldout(&g, &va, window);
            if (!isfinite(val)) goto done;
            if (val < best) {
                if (!checkpoint_save(model_path, &g, NULL)) goto done;
                best = val;
            }
            printf("%d,%.9g,%.9g\n", step, val, best);
        }
    }
    ok = 1;
done:
    if (tr.file) fclose(tr.file);
    if (va.file) fclose(va.file);
    gpt_free(&g);
    if (!ok) fputs("Scale training failed. Check shard IDs, window, paths and memory.\n", stderr);
    return ok;
}

static int smoke(void)
{
    /* GPT-2 encodes 'Hello, world!' as four tokens. EOT closes the example. */
    int ids[] = {15496, 11, 995, 0, 50256};
    Gpt model = {0};
    double before = NAN, after = NAN, norm = NAN;
    int ok = 0;
    if (gpt_init(&model, small, 42) && model.parameter_count == 124439808 &&
        gpt_forward(&model, ids, 4)) {
        before = gpt_backward(&model, ids + 1);
        if (isfinite(before) && gpt_step(&model, rate, 0.1, 1.0, &norm) &&
            gpt_forward(&model, ids, 4)) {
            after = gpt_loss(&model, ids + 1);
            ok = isfinite(after) && isfinite(norm) && norm > 0;
        }
    }
    if (ok) printf("parameters,positions,sequence_tokens,updates,nll_before,nll_after,grad_norm\n"
                   "%zu,1024,4,1,%.9g,%.9g,%.9g\n", model.parameter_count, before, after, norm);
    gpt_free(&model);
    return ok;
}

static int generate_ids(int argc, char **argv, int count)
{
    Gpt g = {0};
    int tokens[1024], n = argc - 4, ok = 0;
    if (n < 1 || n > 1024) return 0;
    for (int i = 0; i < n; ++i)
        if (!number(argv[i + 4], 0, 50256, &tokens[i])) return 0;
    if (!checkpoint_load(argv[2], &g, NULL) || g.config.vocab_size != 50257 ||
        g.config.context != 1024) goto done;
    for (int i = 0; i < n; ++i) printf("%s%d", i ? " " : "", tokens[i]);
    for (int i = 0; i < count; ++i) {
        if (!gpt_forward(&g, tokens, n)) goto done;
        const float *z = gpt_logits(&g, n - 1);
        int best = 0;
        for (int j = 1; j < 50257; ++j) if (z[j] > z[best]) best = j;
        printf(" %d", best);
        if (best == 50256) break;
        if (n == 1024) { memmove(tokens, tokens + 1, 1023 * sizeof(int)); --n; }
        tokens[n++] = best;
    }
    putchar('\n');
    ok = 1;
done:
    gpt_free(&g);
    return ok;
}

int scale_command(int argc, char **argv)
{
    int steps, window, batch = 1, tokens;
    if (argc == 2 && !strcmp(argv[1], "smoke")) return smoke() ? 0 : 1;
    if (argc == 2 && !strcmp(argv[1], "info")) {
        printf("vocab,context,width,heads,layers,parameters\n50257,1024,768,12,12,%zu\n",
               gpt_parameter_count(small));
        return 0;
    }
    if ((argc == 7 || argc == 8) && !strcmp(argv[1], "train") &&
        number(argv[4], 1, 1000000000, &steps) && number(argv[5], 1, 1024, &window) &&
        (argc == 7 || number(argv[7], 1, 1000000, &batch)))
        return train_scale(argv[2], argv[3], steps, window, argv[6], batch) ? 0 : 1;
    if (argc >= 5 && !strcmp(argv[1], "generate") && number(argv[3], 0, 1000000, &tokens))
        return generate_ids(argc, argv, tokens) ? 0 : 1;
    fputs("nablaflow scale info | smoke\n"
          "nablaflow scale train TRAIN.u16 VAL.u16 STEPS WINDOW NEW_MODEL [BATCH]\n"
          "nablaflow scale generate MODEL NEW_TOKENS PROMPT_ID [PROMPT_ID ...]\n", stderr);
    return 1;
}
