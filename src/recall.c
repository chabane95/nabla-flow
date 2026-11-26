#include "checkpoint.h"
#include "gpt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Each line consists of a prompt (preserving its trailing space), a tab,
   and the exact answer including its period. No target is fed to the model. */
int recall_command(int argc, char **argv)
{
    if (argc != 3) {
        fputs("Usage: nablaflow recall MODEL prompts.tsv\n", stderr);
        return EXIT_FAILURE;
    }
    FILE *input = fopen(argv[2], "rb");
    Gpt model = {0};
    int *context = NULL, status = EXIT_FAILURE;
    if (!input || !checkpoint_load(argv[1], &model, NULL) || model.config.vocab_size != 258)
        goto done;
    size_t cap = (size_t)model.config.context;
    context = malloc(cap * sizeof(*context));
    if (!context) goto done;
    char line[2048];
    unsigned total = 0, correct = 0;
    while (fgets(line, sizeof line, input)) {
        if (!strchr(line, '\n') && !feof(input)) goto done;
        line[strcspn(line, "\r\n")] = 0;
        char *answer = strchr(line, '\t');
        if (!answer || !answer[1]) goto done;
        *answer++ = 0;
        size_t bytes = strlen(line), n = 0, start = bytes > cap ? bytes-cap : 0;
        if (bytes < cap) context[n++] = 256;
        for (size_t i = start; i < bytes; ++i) context[n++] = (unsigned char)line[i];
        int matches = 1;
        for (const unsigned char *p = (const unsigned char *)answer; *p; ++p) {
            if (!gpt_forward(&model, context, (int)n)) goto done;
            const float *logits = gpt_logits(&model, (int)n - 1);
            int best = 0;
            for (int k = 1; k < 258; ++k) if (logits[k] > logits[best]) best = k;
            if (best != *p) { matches = 0; break; }
            if (n == cap) {
                memmove(context, context + 1, (cap - 1) * sizeof(*context));
                --n;
            }
            context[n++] = best;
        }
        ++total;
        correct += (unsigned)matches;
    }
    if (!total || ferror(input)) goto done;
    printf("records,exact,exact_rate\n%u,%u,%.9g\n", total, correct, (double)correct / total);
    status = EXIT_SUCCESS;
done:
    if (input) fclose(input);
    free(context);
    gpt_free(&model);
    if (status != EXIT_SUCCESS) fputs("Invalid byte model, prompt file or context length.\n", stderr);
    return status;
}
