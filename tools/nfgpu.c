/* Train GPT-2 Small or Medium on uint16-LE token streams. The GPU
   batch is contiguous [batch,time]. CPU and GPU share a checkpoint layout. */
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif
#include "gpt.h"
#include "gpu.h"
#include "checkpoint.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define create_dir(p) _mkdir(p)
static double seconds(void) {
    LARGE_INTEGER stamp, freq;
    QueryPerformanceCounter(&stamp); QueryPerformanceFrequency(&freq);
    return (double)stamp.QuadPart/(double)freq.QuadPart;
}
#else
#include <time.h>
#include <sys/stat.h>
#define create_dir(p) mkdir(p,0777)
static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#endif

enum { GPT2_VOCAB = 50257, GPT2_CONTEXT = 1024, GPT2_EOT = 50256 };
static const GptConfig small = {GPT2_VOCAB,GPT2_CONTEXT,768,12,12};
static const GptConfig medium = {GPT2_VOCAB,GPT2_CONTEXT,1024,16,24};
typedef struct { uint16_t *ids; size_t count; } Shard;
typedef struct { int *x, *y, batch, window; } Buffers;

static int integer(const char *s, int lo, int hi, int *out)
{
    char *end;
    errno = 0;
    long n = strtol(s,&end,10);
    if (end == s || *end || errno || n < lo || n > hi) return 0;
    *out = (int)n;
    return 1;
}

static int load(Shard *s, const char *path)
{
    FILE *f = fopen(path,"rb");
    long bytes = -1;
    if (f && !fseek(f,0,SEEK_END)) bytes = ftell(f);
    if (bytes >= 2 && bytes%2 == 0 && !fseek(f,0,SEEK_SET))
        s->ids = malloc((size_t)bytes);
    size_t n = bytes > 0 ? (size_t)bytes/2 : 0;
    int ok = f && s->ids && fread(s->ids,2,n,f) == n;
    if (f) fclose(f);
    if (ok) {
        unsigned char *raw = (unsigned char *)s->ids;
        for (size_t i=0;i<n;++i) {
            unsigned id = raw[2*i] | ((unsigned)raw[2*i+1]<<8);
            if (id >= GPT2_VOCAB) { ok=0; break; }
            s->ids[i] = (uint16_t)id;
        }
    }
    if (ok) s->count = n;
    else fprintf(stderr,"Invalid or inaccessible uint16-LE shard: %s\n",path);
    return ok;
}

static int buffers(Buffers *b,int batch,int window)
{
    b->batch=batch; b->window=window;
    b->x=malloc((size_t)batch*window*sizeof(int));
    b->y=malloc((size_t)batch*window*sizeof(int));
    return b->x && b->y;
}

static void copy_pair(Buffers *b, const Shard *s, size_t pos, int row, int length)
{
    for (int t=0;t<length;++t) {
        b->x[row*length+t]=s->ids[pos+t];
        b->y[row*length+t]=s->ids[pos+t+1];
    }
}

static uint64_t random_bits(uint64_t *state)
{
    uint64_t z=(*state+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}

/* Reset attention at each window; count each target after the first exactly once. */
static double evaluate(GPUModel *g, Buffers *buf, const Shard *s)
{
    if (s->count<2) return NAN;
    size_t pos=0; double total=0;
    while (pos < s->count-1) {
        size_t left=s->count-1-pos, full=left/(size_t)buf->window;
        int n=full ? buf->window : (int)left;
        int batch=full ? (int)(full>(size_t)buf->batch?(size_t)buf->batch:full) : 1;
        for (int b=0;b<batch;++b) copy_pair(buf,s,pos+(size_t)b*n,b,n);
        double loss=gpu_pass(g,buf->x,buf->y,batch,n,0);
        if (!isfinite(loss)) return NAN;
        total+=loss*batch*n;
        pos+=(size_t)batch*n;
    }
    return total/(s->count-1);
}

/* The train curve always measures the same 32 windows, including step zero. */
static double probe_train(GPUModel *g, Buffers *buf,const Shard *s)
{
    const int windows=32;
    double total=0;
    for (int first=0;first<windows;) {
        int batch=windows-first;
        if (batch>buf->batch) batch=buf->batch;
        for (int b=0;b<batch;++b) {
            size_t pos=(size_t)((uint64_t)(s->count-1-buf->window)*(first+b)/(windows-1));
            copy_pair(buf,s,pos,b,buf->window);
        }
        total+=batch*gpu_pass(g,buf->x,buf->y,batch,buf->window,0);
        first+=batch;
    }
    return total/windows;
}

static int sample_token(const float *z,uint64_t *rng)
{
    enum { k=40 };
    int order[k], count=0;
    for (int v=0;v<GPT2_VOCAB;++v) {
        if (!isfinite(z[v])) return -1;
        if (count==k && z[v]<=z[order[k-1]]) continue;
        int i=count<k ? count++ : k-1;
        while (i && z[order[i-1]]<z[v]) { order[i]=order[i-1]; --i; }
        order[i]=v;
    }
    double sums[k], total=0;
    for (int j=0;j<k;++j) {
        total+=exp(((double)z[order[j]]-z[order[0]])/0.8);
        sums[j]=total;
    }
    double draw=(double)(random_bits(rng)>>11)*(1.0/9007199254740992.0)*total;
    for (int i=0;i<k-1;++i) if (draw<sums[i]) return order[i];
    return order[k-1];
}

static int sample(GPUModel *g,const Shard *prompt,int cap,int count,FILE *f)
{
    int ctx[GPT2_CONTEXT];
    float *logits=malloc((size_t)GPT2_VOCAB*sizeof(*logits));
    int ok=logits && prompt->count && prompt->count<=(size_t)cap;
    int n=ok ? (int)prompt->count : 0;
    for (int i=0;i<n;++i) ctx[i]=prompt->ids[i];
    uint64_t state=42;
    for (int j=0;ok && j<count;++j) {
        ok=isfinite(gpu_pass(g,ctx,NULL,1,n,0)) && gpu_last_logits(g,0,n-1,logits);
        int token=ok ? sample_token(logits,&state) : -1;
        if (token<0) { ok=0; break; }
        fprintf(f,"%s%d",j?" ":"",token);
        if (token==GPT2_EOT) break;
        if (n==cap) { memmove(ctx,ctx+1,(size_t)(cap-1)*sizeof(int)); --n; }
        ctx[n++]=token;
    }
    free(logits);
    return ok && !ferror(f);
}

static char *join(const char *dir,const char *name)
{
    size_t len=strlen(dir)+strlen(name)+2;
    char *path=malloc(len);
    if (path) snprintf(path,len,"%s/%s",dir,name);
    return path;
}

static double learning_rate(int step,int steps)
{
    int warmup=steps/10 < 100 ? steps/10 : 100;
    if (warmup<1) warmup=1;
    if (step<=warmup) return 0.0003*step/warmup;
    double progress=(double)(step-warmup)/(steps-warmup);
    return 0.00003+0.00027*0.5*(1+cos(3.141592653589793*progress));
}

static int train(int argc,char **argv)
{
    int steps,window,batch,interval,seed;
    if ((argc!=11 && argc!=12) || !integer(argv[5],1,10000000,&steps) ||
        !integer(argv[6],1,GPT2_CONTEXT,&window) ||
        !integer(argv[7],1,64,&batch) || !integer(argv[8],1,10000000,&interval) ||
        !integer(argv[9],0,INT_MAX,&seed)) return 0;
    const char *name=argc==12 ? argv[11] : "small";
    GptConfig config;
    if (!strcmp(name,"small")) config=small;
    else if (!strcmp(name,"medium")) config=medium;
    else return 0;
    Shard tr={0},va={0},pr={0};
    Gpt host={0}; GPUModel *g=NULL; Buffers buf={0};
    FILE *history=NULL,*samples=NULL,*batches=NULL,*memory=NULL;
    char *weights=NULL,*histpath=NULL,*samplepath=NULL,*batchpath=NULL;
    int ok=0, best_step=0; double best=HUGE_VAL, norm=0, lr=0;
    if (!strcmp(argv[2],argv[3]) || !load(&tr,argv[2]) || !load(&va,argv[3]) || !load(&pr,argv[10]) ||
        tr.count<=(size_t)window || va.count<2 || pr.count>(size_t)window ||
        !buffers(&buf,batch,window) || create_dir(argv[4]) != 0) goto done;
    weights=join(argv[4],"best.nfg"); histpath=join(argv[4],"history.csv");
    samplepath=join(argv[4],"samples.csv"); batchpath=join(argv[4],"batches.csv");
    if (config.width==medium.width) {
        char *path=join(argv[4],"memory.csv");
        if (path) memory=fopen(path,"w");
        free(path);
        if (!memory) goto done;
    }
    if (!weights || !histpath || !samplepath || !batchpath ||
        !(history=fopen(histpath,"w")) || !(samples=fopen(samplepath,"w")) ||
        !(batches=fopen(batchpath,"w")) || !gpt_init(&host,config,(uint32_t)seed) ||
        !(g=gpu_create_profiled(&host,batch,window,memory))) goto done;
    printf("GPT-2 %s | %zu parameters | width %d | heads %d | blocks %d\n",
           name,host.parameter_count,config.width,config.heads,config.layers);
    fputs("step,tokens,train_nll,val_nll,lr,grad_norm,elapsed_seconds\n",history);
    fputs("step,generated_ids\n",samples);
    fputs("step,tokens,batch_nll,lr,grad_norm\n",batches);
    uint64_t rng=(uint32_t)seed;
    double start=seconds();
    for (int step=0; step<=steps; ++step) {
        uint64_t tokens=(uint64_t)step*batch*window;
        if (step) {
            for (int b=0;b<batch;++b) {
                size_t pos=(size_t)(random_bits(&rng)%(tr.count-window));
                copy_pair(&buf,&tr,pos,b,window);
            }
            double loss=gpu_pass(g,buf.x,buf.y,batch,window,1);
            int measure=memory && (step==1 || step%interval==0 || step==steps);
            lr=learning_rate(step,steps);
            if (!isfinite(loss) || (measure && !gpu_measure_memory(g,"after_backward",step)) ||
                !gpu_update(g,lr,0.1,1.0,&norm) || !isfinite(norm) ||
                (measure && !gpu_measure_memory(g,"after_update",step))) goto done;
            fprintf(batches,"%d,%llu,%.9g,%.9g,%.9g\n",step,(unsigned long long)tokens,loss,lr,norm);
            if (step%10==0 || step==steps) {
                fflush(batches);
                if (ferror(batches)) goto done;
                printf("step %d/%d | batch NLL %.4f | %.1f s\n",step,steps,loss,seconds()-start);
                fflush(stdout);
            }
        }
        if (step%interval && step!=steps) continue;
        double tr_loss=probe_train(g,&buf,&tr), val=evaluate(g,&buf,&va);
        if (!isfinite(tr_loss) || !isfinite(val) || !gpu_measure_memory(g,"after_validation",step))
            goto done;
        fprintf(history,"%d,%llu,%.9g,%.9g,%.9g,%.9g,%.3f\n",step,(unsigned long long)tokens,
                tr_loss,val,lr,norm,seconds()-start);
        fflush(history);
        if (ferror(history)) goto done;
        if (val<best) {
            if (!gpu_export(g,&host) || !checkpoint_save(weights,&host,NULL)) goto done;
            best=val; best_step=step;
        }
        printf("validation %d | train %.4f | val %.4f | selected %d\n",step,tr_loss,val,best_step);
        fflush(stdout);
        /* Capture each measured state so the figures and text can be compared
           without changing the seed or hand-picking a favorable generation. */
        fprintf(samples,"%d,",step);
        if (!sample(g,&pr,window,64,samples)) goto done;
        fputc('\n',samples); fflush(samples);
        if (ferror(samples) || !gpu_measure_memory(g,"after_sampling",step)) goto done;
    }
    ok=1;
done:
    if (memory && fclose(memory)) ok=0;
    if (samples && fclose(samples)) ok=0;
    if (history && fclose(history)) ok=0;
    if (batches && fclose(batches)) ok=0;
    gpu_destroy(g); gpt_free(&host); free(buf.x); free(buf.y);
    free(tr.ids); free(va.ids); free(pr.ids); free(weights); free(histpath); free(samplepath); free(batchpath);
    if (!ok) fputs("Training failed. Check input shards, unused output path and GPU runtime.\n",stderr);
    return ok;
}

static int matches_preset(GptConfig c,GptConfig preset)
{
    return c.vocab_size==preset.vocab_size && c.context==preset.context &&
           c.width==preset.width && c.heads==preset.heads && c.layers==preset.layers;
}

static int inference(int argc,char **argv,int generation)
{
    int n,window;
    if (argc!=6 || !integer(argv[4],1,generation?10000:64,&n) ||
        !integer(argv[5],1,GPT2_CONTEXT,&window)) return 0;
    int batch=generation ? 1 : n;
    Shard s={0}; Buffers buf={0}; Gpt host={0}; GPUModel *g=NULL;
    int ok=load(&s,argv[3]) && checkpoint_load(argv[2],&host,NULL) &&
           (matches_preset(host.config,small) || matches_preset(host.config,medium)) &&
           (g=gpu_create(&host,batch,window)) && buffers(&buf,batch,window);
    if (ok && generation) {
        ok=sample(g,&s,window,n,stdout); putchar('\n');
    } else if (ok) {
        double nll=evaluate(g,&buf,&s); ok=isfinite(nll);
        if (ok) printf("targets,nll,perplexity\n%zu,%.9g,%.9g\n",s.count-1,nll,exp(nll));
    }
    gpu_destroy(g); gpt_free(&host); free(s.ids); free(buf.x); free(buf.y);
    return ok;
}

/* Context-free reference on the identical BPE alphabet and target positions. */
static int unigram(const char *training,const char *test)
{
    Shard tr={0}, te={0};
    size_t *counts=calloc(GPT2_VOCAB,sizeof(*counts));
    int ok=counts && load(&tr,training) && load(&te,test) && tr.count>1 && te.count>1;
    if (ok) {
        for (size_t i=1;i<tr.count;++i) ++counts[tr.ids[i]];
        double denom=(double)(tr.count-1)+0.1*GPT2_VOCAB;
        double nll=0;
        for (size_t i=1;i<te.count;++i)
            nll-=log(((double)counts[te.ids[i]]+0.1)/denom);
        nll/=(te.count-1);
        printf("targets,nll,perplexity\n%zu,%.9g,%.9g\n",te.count-1,nll,exp(nll));
    }
    free(counts); free(tr.ids); free(te.ids);
    return ok;
}

int main(int argc,char **argv)
{
    if (argc==2 && !strcmp(argv[1],"check")) return gpu_selftest()?0:1;
    if (argc>1 && !strcmp(argv[1],"train") && train(argc,argv)) return 0;
    if (argc>1 && !strcmp(argv[1],"eval") && inference(argc,argv,0)) return 0;
    if (argc>1 && !strcmp(argv[1],"sample") && inference(argc,argv,1)) return 0;
    if (argc==4 && !strcmp(argv[1],"unigram") && unigram(argv[2],argv[3])) return 0;
    fputs("nfgpu check\nnfgpu train TRAIN.u16 VAL.u16 NEW_RUN STEPS WINDOW BATCH EVAL_EVERY SEED PROMPT.u16 [small|medium]\n"
          "nfgpu eval MODEL.nfg TEST.u16 BATCH WINDOW\nnfgpu sample MODEL.nfg PROMPT.u16 TOKENS WINDOW\n"
          "nfgpu unigram TRAIN.u16 TEST.u16\n",stderr);
    return 1;
}
