#include "checkpoint.h"
#include "data.h"
#include "gpt.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define make_dir(path) _mkdir(path)
#else
#include <sys/stat.h>
#define make_dir(path) mkdir(path, 0777)
#endif

int scale_command(int argc, char **argv);
int recall_command(int argc, char **argv);

static const char *usage =
    "nablaflow train [--train data/train.txt --val data/val.txt --out run]\n"
    "                [--steps 600 --batch 8 --ctx 32 --width 32 --heads 4 --layers 1]\n"
    "                [--lr 0.002 --eval-every 25 --seed 42]\n"
    "nablaflow sample --model FILE [--prompt TEXT --tokens 80 --temperature 0.7 --top-k 8 --seed 7]\n"
    "nablaflow inspect --model FILE [--prompt TEXT --layer 0 --head 0]\n"
    "nablaflow eval --model FILE --data TEXTFILE [--train TEXTFILE]\n"
    "nablaflow scale ...\n"
    "nablaflow recall MODEL data/recall.tsv\n"
    "nablaflow check\n";

static uint64_t random_bits(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void shuffle(size_t order[], size_t count, uint64_t *state)
{
    for (size_t i = count; i > 1; --i) {
        uint64_t limit = (uint64_t)(-(uint64_t)i) % i, bits;
        do { bits = random_bits(state); } while (bits < limit);
        size_t j = (size_t)(bits % i), tmp = order[i-1];
        order[i-1] = order[j];
        order[j] = tmp;
    }
}

static int integer(const char *s, int min, int max, int *out)
{
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (end == s || *end || errno || n < min || n > max)
        return 0;
    *out = (int)n;
    return 1;
}

static int seed_value(const char *s, uint32_t *out)
{
    if (*s < '0' || *s > '9') return 0;
    char *end;
    errno = 0;
    unsigned long n = strtoul(s, &end, 10);
    if (*end || errno || n > UINT32_MAX) return 0;
    *out = (uint32_t)n;
    return 1;
}

static int real(const char *s, double *out)
{
    char *end;
    errno = 0;
    double n = strtod(s, &end);
    if (s == end || *end || errno || !isfinite(n)) return 0;
    *out = n;
    return 1;
}

typedef struct {
    const char *train, *val, *out;
    int steps, batch, eval_every;
    uint32_t seed;
    double lr;
    GptConfig config;
} TrainOptions;

static int train_options(TrainOptions *p, int argc, char **argv)
{
    *p = (TrainOptions){"data/train.txt","data/val.txt","run",600,8,25,42,0.002,
                        {NF_VOCAB,32,32,4,1}};
    for (int i = 2; i < argc; i += 2) {
        const char *k = argv[i], *v = argv[i+1];
        if (!v || !*v) return 0;
        if (!strcmp(k,"--train")) p->train = v;
        else if (!strcmp(k,"--val")) p->val = v;
        else if (!strcmp(k,"--out")) p->out = v;
        else if (!strcmp(k,"--steps") && integer(v,1,INT_MAX-1,&p->steps)) continue;
        else if (!strcmp(k,"--batch") && integer(v,1,INT_MAX-1,&p->batch)) continue;
        else if (!strcmp(k,"--eval-every") && integer(v,1,INT_MAX-1,&p->eval_every)) continue;
        else if (!strcmp(k,"--ctx") && integer(v,1,2048,&p->config.context)) continue;
        else if (!strcmp(k,"--width") && integer(v,2,768,&p->config.width)) continue;
        else if (!strcmp(k,"--heads") && integer(v,1,768,&p->config.heads)) continue;
        else if (!strcmp(k,"--layers") && integer(v,1,24,&p->config.layers)) continue;
        else if (!strcmp(k,"--seed") && seed_value(v,&p->seed)) continue;
        else if (!strcmp(k,"--lr") && real(v,&p->lr) && p->lr > 0) continue;
        else return 0;
    }
    return strcmp(p->train,p->val) != 0 && gpt_parameter_count(p->config) != 0;
}

static char *join(const char *dir, const char *file)
{
    size_t n = strlen(dir) + strlen(file) + 2;
    char *path = malloc(n);
    if (path) snprintf(path, n, "%s/%s", dir, file);
    return path;
}

static double evaluate(Gpt *g, const Corpus *c, size_t blocks, size_t *tokens)
{
    double total = 0;
    size_t n = 0;
    for (size_t i = 0; i < blocks; ++i) {
        /* For the training probe take fixed, evenly spaced blocks. */
        size_t index = blocks == c->block_count ? i :
                       (i * (c->block_count / blocks) + i * (c->block_count % blocks) / blocks);
        Block b = c->blocks[index];
        const int *x = c->tokens + b.start;
        if (!gpt_forward(g, x, b.length))
            return NAN;
        total += b.length * gpt_loss(g,x+1);
        n += (size_t)b.length;
    }
    if (tokens) *tokens = n;
    return total / (double)n;
}

static double learning_rate(int step, int steps, double peak)
{
    int warmup = steps / 20;
    if (warmup < 1) warmup = 1;
    if (warmup > 20) warmup = 20;
    if (step <= warmup) return peak * step / warmup;
    double phase = 3.14159265358979323846 * (step - warmup) / (steps - warmup);
    return peak * (0.1 + 0.45 * (1 + cos(phase)));
}

static int close_file(FILE *f)
{
    int ok = !ferror(f);
    return fclose(f) == 0 && ok;
}

static int train(const TrainOptions *o)
{
    Corpus tr = {0}, va = {0};
    Gpt g = {0};
    int ok = 0, best_step = 0;
    uint64_t rng = o->seed ^ UINT32_C(0xa842bc91);
    FILE *hist = NULL, *summary = NULL;
    size_t *order = NULL, cursor = 0;
    float *best = NULL;
    char *mp = join(o->out,"model.nfg"), *hp = join(o->out,"history.csv"), *sp = join(o->out,"summary.csv");
    if (!mp || !hp || !sp || !corpus_load(&tr,o->train,o->config.context) ||
        !corpus_load(&va,o->val,o->config.context) || !gpt_init(&g,o->config,o->seed))
        goto done;
    double baseline = bigram_nll(&tr,&va);
    double best_val = HUGE_VAL, initial = 0, norm = 0, lr = 0;
    size_t train_probe = tr.block_count < 64 ? tr.block_count : 64, probe_tokens = 0;
    best = malloc(g.parameter_count * sizeof(*best));
    order = malloc(tr.block_count * sizeof(*order));
    if (!isfinite(baseline) || !best || !order) goto done;
    if (make_dir(o->out)) {
        perror(o->out);
        fputs("Choose a new output directory under an existing parent.\n",stderr);
        goto done;
    }
    hist = fopen(hp,"w");
    if (!hist) goto done;
    fputs("step,train_nll,val_nll,grad_norm,learning_rate\n",hist);
    for (size_t i = 0; i < tr.block_count; ++i) order[i] = i;
    cursor = tr.block_count;
    for (int step = 0; step <= o->steps; ++step) {
        if (step) {
            for (int b = 0; b < o->batch; ++b) {
                if (cursor == tr.block_count) {
                    shuffle(order,tr.block_count,&rng);
                    cursor = 0;
                }
                Block block = tr.blocks[order[cursor++]];
                const int *x = tr.tokens + block.start;
                if (!gpt_forward(&g,x,block.length) || !isfinite(gpt_backward(&g,x+1)))
                    goto done;
            }
            lr = learning_rate(step,o->steps,o->lr);
            if (!gpt_step(&g,lr,0.01,1.0,&norm))
                goto done;
        }
        if (step % o->eval_every != 0 && step != o->steps)
            continue;
        double t = evaluate(&g,&tr,train_probe,&probe_tokens), v = evaluate(&g,&va,va.block_count,NULL);
        if (!isfinite(t) || !isfinite(v) || !isfinite(norm) ||
            fprintf(hist,"%d,%.12g,%.12g,%.12g,%.12g\n",step,t,v,norm,lr)<0 || fflush(hist))
            goto done;
        if (!step) initial = v;
        if (v < best_val) {
            best_val = v;
            best_step = step;
            memcpy(best,g.parameters,g.parameter_count * sizeof(*best));
        }
        printf("step %d/%d  train %.4f  val %.4f\n",step,o->steps,t,v);
        fflush(stdout);
    }
    memcpy(g.parameters,best,g.parameter_count * sizeof(*best));
    if (!checkpoint_save(mp,&g,tr.seen)) {
        fprintf(stderr,"Could not save %s\n",mp);
        goto done;
    }
    summary = fopen(sp,"w");
    if (!summary) goto done;
    fputs("seed,steps,best_step,train_records,val_records,train_tokens,val_tokens,parameters,"
          "initial_val_nll,best_val_nll,bigram_val_nll,bigram_val_ppl,"
          "context,width,heads,layers,batch,peak_lr,eval_every,train_probe_blocks,train_probe_tokens\n",summary);
    fprintf(summary,"%u,%d,%d,%zu,%zu,%zu,%zu,%zu,%.12g,%.12g,%.12g,%.12g,"
                    "%d,%d,%d,%d,%d,%.12g,%d,%zu,%zu\n",
            (unsigned)o->seed,o->steps,best_step,tr.records,va.records,tr.targets,va.targets,g.parameter_count,
            initial,best_val,baseline,exp(baseline),o->config.context,o->config.width,o->config.heads,
            o->config.layers,o->batch,o->lr,o->eval_every,train_probe,probe_tokens);
    printf("selected step %d | validation NLL %.5f | bigram NLL %.5f\n",best_step,best_val,baseline);
    ok = 1;
done:
    if (hist && !close_file(hist)) ok = 0;
    if (summary && !close_file(summary)) ok = 0;
    if (!ok) fputs("Training failed. Check the inputs, model configuration and output directory.\n",stderr);
    free(mp); free(hp); free(sp); free(best); free(order);
    gpt_free(&g); corpus_free(&tr); corpus_free(&va);
    return ok;
}

typedef struct {
    const char *model, *data, *train, *prompt;
    int tokens, top_k, layer, head;
    double temperature;
    uint32_t seed;
} ReadOptions;

static int read_options(ReadOptions *o, int argc, char **argv)
{
    *o = (ReadOptions){NULL,NULL,NULL,"",80,8,0,0,0.7,7};
    int sampling = !strcmp(argv[1],"sample"), inspection = !strcmp(argv[1],"inspect");
    for (int i = 2; i < argc; i += 2) {
        const char *k = argv[i], *v = argv[i+1];
        if (!v) return 0;
        if (!strcmp(k,"--model") && *v) o->model = v;
        else if ((sampling || inspection) && !strcmp(k,"--prompt")) o->prompt = v;
        else if (sampling && !strcmp(k,"--tokens") && integer(v,0,INT_MAX-1,&o->tokens)) continue;
        else if (sampling && !strcmp(k,"--top-k") && integer(v,0,NF_VOCAB,&o->top_k)) continue;
        else if (sampling && !strcmp(k,"--temperature") && real(v,&o->temperature) && o->temperature >= 0) continue;
        else if (sampling && !strcmp(k,"--seed") && seed_value(v,&o->seed)) continue;
        else if (inspection && !strcmp(k,"--layer") && integer(v,0,INT_MAX-1,&o->layer)) continue;
        else if (inspection && !strcmp(k,"--head") && integer(v,0,INT_MAX-1,&o->head)) continue;
        else if (!sampling && !inspection && !strcmp(k,"--data") && *v) o->data = v;
        else if (!sampling && !inspection && !strcmp(k,"--train") && *v) o->train = v;
        else return 0;
    }
    return o->model && (sampling || inspection || o->data);
}

static int encode_prompt(const char *text, int context, int *ids)
{
    const unsigned char *bytes = (const unsigned char *)text;
    size_t n = strlen(text);
    Utf8State state = {0};
    for (size_t i = 0; i < n; ++i)
        if (!utf8_byte(&state,bytes[i]))
            return 0;
    if (state.left) return 0;
    int length = 0;
    if (n < (size_t)context) ids[length++] = NF_BOS;
    size_t first = n > (size_t)context ? n - (size_t)context : 0;
    for (size_t i = first; i < n; ++i)
        ids[length++] = bytes[i];
    return length;
}

static int inspect(Gpt *g, const ReadOptions *o, int *ids, int n)
{
    if (o->layer >= g->config.layers || o->head >= g->config.heads || !gpt_forward(g,ids,n))
        return 0;
    float *row = malloc((size_t)n * sizeof(*row));
    if (!row) return 0;
    puts("query_pos,key_pos,query_token,key_token,weight");
    int ok = 1;
    for (int q = 0; ok && q < n; ++q) {
        ok = gpt_attention_row(g,o->layer,o->head,q,row);
        if (ok)
            for (int k = 0; k < n; ++k)
                printf("%d,%d,%d,%d,%.9g\n",q,k,ids[q],ids[k],(double)row[k]);
    }
    free(row);
    return ok && !ferror(stdout);
}

/* Sampling is restricted to observed printable bytes, valid UTF-8 transitions
   and EOS at scalar boundaries. Loss evaluation always uses all 258 logits. */
static int choose(const float *logits, const unsigned char seen[256], Utf8State state,
                  int top_k, double temperature, uint64_t *rng)
{
    int order[NF_VOCAB], n = 0;
    for (int token = 0; token < NF_VOCAB; ++token) {
        Utf8State next = state;
        if (token == NF_BOS) continue;
        if (token == NF_EOS) {
            if (state.left) continue;
        } else if (!seen[token] || (token < 32 && token != '\t') || token == 127 ||
                   !utf8_byte(&next,(unsigned)token)) {
            continue;
        }
        if (!isfinite(logits[token])) return -1;
        int j = n++;
        while (j > 0 && logits[order[j-1]] < logits[token]) {
            order[j] = order[j-1];
            --j;
        }
        order[j] = token;
    }
    if (!n) return -1;
    if (temperature == 0) return order[0];
    if (top_k > 0 && n > top_k) n = top_k;
    double cumulative[NF_VOCAB], sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += exp(((double)logits[order[i]] - logits[order[0]]) / temperature);
        cumulative[i] = sum;
    }
    double u = (double)(random_bits(rng) >> 11) / 9007199254740992.0 * sum;
    for (int i = 0; i < n-1; ++i)
        if (u < cumulative[i]) return order[i];
    return order[n-1];
}

static int sample(Gpt *g, const ReadOptions *o, int *ids, int n, const unsigned char seen[256])
{
    uint64_t rng = o->seed;
    Utf8State state = {0};
    unsigned char pending[4];
    int buffered = 0;
    fputs(o->prompt,stdout);
    for (int i = 0; i < o->tokens; ++i) {
        if (!gpt_forward(g,ids,n)) return 0;
        int token = choose(gpt_logits(g,n-1),seen,state,o->top_k,o->temperature,&rng);
        if (token < 0) return 0;
        if (token == NF_EOS) break;
        if (!utf8_byte(&state,(unsigned)token) || buffered >= 4) return 0;
        pending[buffered++] = (unsigned char)token;
        if (!state.left) {
            if (fwrite(pending,1,(size_t)buffered,stdout) != (size_t)buffered) return 0;
            buffered = 0;
        }
        if (n == g->config.context) {
            memmove(ids,ids+1,(size_t)(n-1) * sizeof(*ids));
            --n;
        }
        ids[n++] = token;
    }
    return putchar('\n') != EOF && !ferror(stdout);
}

static int eval_cli(Gpt *g, const ReadOptions *o)
{
    Corpus test = {0}, tr = {0};
    int ok = corpus_load(&test,o->data,g->config.context) &&
             (!o->train || corpus_load(&tr,o->train,g->config.context));
    if (ok) {
        double nll = evaluate(g,&test,test.block_count,NULL), bl = o->train ? bigram_nll(&tr,&test) : 0;
        ok = isfinite(nll) && isfinite(bl);
        if (ok) {
            printf("records,tokens,nll,perplexity%s\n",o->train ? ",bigram_nll,bigram_perplexity" : "");
            printf("%zu,%zu,%.12g,%.12g",test.records,test.targets,nll,exp(nll));
            if (o->train) printf(",%.12g,%.12g",bl,exp(bl));
            putchar('\n');
        }
    }
    corpus_free(&test); corpus_free(&tr);
    return ok && !ferror(stdout);
}

int main(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1],"--help") || !strcmp(argv[1],"-h")) {
        fputs(usage,argc < 2 ? stderr : stdout);
        return argc < 2 ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    if (!strcmp(argv[1],"scale"))
        return scale_command(argc-1,argv+1);
    if (!strcmp(argv[1],"recall"))
        return recall_command(argc-1,argv+1);
    if (!strcmp(argv[1],"check") && argc == 2)
        return data_selftest() && gpt_selftest() && checkpoint_selftest() ? EXIT_SUCCESS : EXIT_FAILURE;
    if (!strcmp(argv[1],"train")) {
        TrainOptions o;
        if (train_options(&o,argc,argv)) return train(&o) ? EXIT_SUCCESS : EXIT_FAILURE;
    } else if (!strcmp(argv[1],"sample") || !strcmp(argv[1],"inspect") || !strcmp(argv[1],"eval")) {
        ReadOptions o;
        if (read_options(&o,argc,argv)) {
            Gpt g = {0};
            unsigned char seen[256];
            if (!checkpoint_load(o.model,&g,seen) || g.config.vocab_size != NF_VOCAB) {
                fprintf(stderr,"Cannot load a byte-model checkpoint from %s\n",o.model);
                gpt_free(&g);
                return EXIT_FAILURE;
            }
            int ok = 0;
            if (o.data) {
                ok = eval_cli(&g,&o);
            } else {
                int *ids = malloc((size_t)g.config.context * sizeof(*ids));
                int n = ids ? encode_prompt(o.prompt,g.config.context,ids) : 0;
                if (n)
                    ok = !strcmp(argv[1],"sample") ? sample(&g,&o,ids,n,seen) : inspect(&g,&o,ids,n);
                free(ids);
            }
            gpt_free(&g);
            if (!ok) fputs("Evaluation failed. Check the UTF-8 input and layer/head indices.\n",stderr);
            return ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    fputs(usage,stderr);
    return EXIT_FAILURE;
}
