/* Optional CUDA backend. The ordinary CPU build has no CUDA dependencies.
   Uses the CUDA Driver API, NVRTC and cuBLAS via their stable C entry points.
   The host and the device exchange only tokens, metrics and requested weights. */
#include "gpu.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HMODULE Library;
#define libopen(name) LoadLibraryA(name)
#define DRIVER "nvcuda.dll"
#define NVRTC "nvrtc64_120_0.dll"
#define BLAS "cublas64_12.dll"
#else
#include <dlfcn.h>
typedef void *Library;
#define libopen(name) dlopen(name,RTLD_NOW)
#define DRIVER "libcuda.so.1"
#define NVRTC "libnvrtc.so.12"
#define BLAS "libcublas.so.12"
#endif

/* Device pointer in the 64-bit CUDA driver ABI. No device pointer is ever
   dereferenced by the host. */
typedef uint64_t Dev;
typedef struct {
    Library driver, rtc, blas;
    int (*init)(unsigned);
    int (*device)(int *,int);
    int (*attribute)(int *,int,int);
    int (*retain)(void **,int);
    int (*release)(int);
    int (*current)(void *);
    int (*alloc)(Dev *,size_t);
    int (*dealloc)(Dev);
    int (*upload)(Dev,const void *,size_t);
    int (*download)(void *,Dev,size_t);
    int (*zero)(Dev,unsigned,size_t);
    int (*load)(void **,const void *);
    int (*unload)(void *);
    int (*function)(void **,void *,const char *);
    int (*launch)(void *,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void *,void **,void **);
    int (*error)(int,const char **);
    int (*make_program)(void **,const char *,const char *,int,const char **,const char **);
    int (*compile)(void *,int,const char **);
    int (*log_size)(void *,size_t *);
    int (*log)(void *,char *);
    int (*ptx_size)(void *,size_t *);
    int (*ptx)(void *,char *);
    int (*destroy_program)(void **);
    int (*make_blas)(void **);
    int (*destroy_blas)(void *);
    int (*gemm)(void *,int,int,int,int,int,const void *,const void *,int,int,
                const void *,int,int,const void *,void *,int,int,int,int);
    int (*nrm2)(void *,int,const float *,int,float *);
} Cuda;

static Cuda api;

static int symbol(Library lib,const char *name,void *target,size_t bytes)
{
#ifdef _WIN32
    FARPROC fn=GetProcAddress(lib,name);
#else
    void *fn=dlsym(lib,name);
#endif
    if (!fn || sizeof(fn)!=bytes) {
        fprintf(stderr,"Missing CUDA entry point %s\n",name);
        return 0;
    }
    memcpy(target,&fn,bytes);
    return 1;
}

#define FIND(lib,field,name) if (!symbol(api.lib,name,&api.field,sizeof(api.field))) return 0
static int load_api(void)
{
    static int ready;
    if (ready) return 1;
    const char *rtc=getenv("NFGPU_NVRTC_DLL");
    api.driver=libopen(DRIVER); api.rtc=libopen(rtc?rtc:NVRTC); api.blas=libopen(BLAS);
    if (!api.driver || !api.rtc || !api.blas) {
        fprintf(stderr,"Cannot load %s, %s or %s. Check CUDA 12 runtime search paths.\n",
                DRIVER,rtc?rtc:NVRTC,BLAS);
        return 0;
    }
    FIND(driver,init,"cuInit"); FIND(driver,device,"cuDeviceGet");
    FIND(driver,attribute,"cuDeviceGetAttribute"); FIND(driver,retain,"cuDevicePrimaryCtxRetain");
    FIND(driver,release,"cuDevicePrimaryCtxRelease_v2"); FIND(driver,current,"cuCtxSetCurrent");
    FIND(driver,alloc,"cuMemAlloc_v2"); FIND(driver,dealloc,"cuMemFree_v2");
    FIND(driver,upload,"cuMemcpyHtoD_v2"); FIND(driver,download,"cuMemcpyDtoH_v2");
    FIND(driver,zero,"cuMemsetD32_v2"); FIND(driver,load,"cuModuleLoadData");
    FIND(driver,unload,"cuModuleUnload"); FIND(driver,function,"cuModuleGetFunction");
    FIND(driver,launch,"cuLaunchKernel"); FIND(driver,error,"cuGetErrorString");
    FIND(rtc,make_program,"nvrtcCreateProgram"); FIND(rtc,compile,"nvrtcCompileProgram");
    FIND(rtc,log_size,"nvrtcGetProgramLogSize"); FIND(rtc,log,"nvrtcGetProgramLog");
    FIND(rtc,ptx_size,"nvrtcGetPTXSize"); FIND(rtc,ptx,"nvrtcGetPTX");
    FIND(rtc,destroy_program,"nvrtcDestroyProgram");
    FIND(blas,make_blas,"cublasCreate_v2"); FIND(blas,destroy_blas,"cublasDestroy_v2");
    FIND(blas,gemm,"cublasGemmEx"); FIND(blas,nrm2,"cublasSnrm2_v2");
    ready=1;
    return 1;
}
#undef FIND

static int driver_ok(int result,const char *call)
{
    if (!result) return 1;
    const char *description=NULL;
    if (api.error) api.error(result,&description);
    fprintf(stderr,"%s: CUDA error %d (%s)\n",call,result,description?description:"unknown");
    return 0;
}
#define CUDA(call) driver_ok(api.call,#call)

enum Kernel {
    EMBED, EMBED_BACK, NORM, NORM_BACK, ADD, BIAS_BACK, GELU, ATTENTION,
    SCORE_BACK, QKV_BACK, XENT, ADAM, N_KERNELS
};
static const char *const names[N_KERNELS]={"embedding","embedding_back","layernorm","layernorm_back",
    "add_vector","bias_back","gelu_kernel","attention","attention_scores_back","attention_qkv_back",
    "cross_entropy","adamw"};

typedef struct {
    Dev norm1,qkv,attention,mix,mid,norm2,pre,gelu,mean1,inv1,mean2,inv2;
} Layer;

struct GPUModel {
    GptConfig c;
    int device,max_b,max_t,last_b,last_t,has_logits,compute;
    size_t count,matrices,stride;
    uint64_t steps,tokens;
    void *context,*module,*blas,*kernel[N_KERNELS];
    Dev training,arena,weights,grad,m,v,input,target,loss,hidden,final,mean,inv,logits;
    Dev dx,dn,dmix,dqkv,dff,dscores;
    Layer *cache;
    float *host_loss;
};

static char *source_file(void)
{
    const char *override=getenv("NFGPU_KERNELS");
    FILE *f=fopen(override?override:"src/gpu_kernels.cu","rb");
    if (!f && !override) f=fopen("gpu_kernels.cu","rb");
    if (!f) {
        fputs("Cannot find src/gpu_kernels.cu. Set NFGPU_KERNELS to its path.\n",stderr);
        return NULL;
    }
    long n=-1;
    if (!fseek(f,0,SEEK_END)) n=ftell(f);
    char *result=n>0?malloc((size_t)n+1):NULL;
    if (result) {
        if (fseek(f,0,SEEK_SET) || fread(result,1,(size_t)n,f)!=(size_t)n) {
            free(result); result=NULL;
        } else result[n]=0;
    }
    fclose(f);
    return result;
}

static int compile_kernels(GPUModel *g)
{
    char *source=source_file(), *ptx=NULL;
    void *program=NULL;
    int major=0,minor=0,ok=0;
    if (!source || !CUDA(attribute(&major,75,g->device)) || !CUDA(attribute(&minor,76,g->device)))
        goto done;
    /* PTX is forward-compatible. CUDA 12 supports compute_80 on all Ampere+
       devices used here, even for architectures newer than the NVRTC release. */
    if (major<8) { fputs("This backend requires compute capability 8.0+.\n",stderr); goto done; }
    const char *options[]={"--gpu-architecture=compute_80","--std=c++14"};
    if (api.make_program(&program,source,"gpu_kernels.cu",0,NULL,NULL)) goto done;
    int status=api.compile(program,2,options);
    if (status) {
        size_t n=0;
        api.log_size(program,&n);
        char *log=n?malloc(n):NULL;
        if (log) { api.log(program,log); fputs(log,stderr); free(log); }
        fprintf(stderr,"NVRTC compile error %d\n",status);
        goto done;
    }
    size_t n=0;
    if (api.ptx_size(program,&n) || !(ptx=malloc(n)) || api.ptx(program,ptx) ||
        !CUDA(load(&g->module,ptx))) goto done;
    for (int i=0;i<N_KERNELS;i++)
        if (!CUDA(function(&g->kernel[i],g->module,names[i]))) goto done;
    ok=1;
done:
    if (program) api.destroy_program(&program);
    free(source); free(ptx);
    return ok;
}

static Dev at(Dev base,size_t floats) { return base+sizeof(float)*floats; }
static Dev take(Dev *cursor,size_t floats)
{
    Dev result=*cursor;
    *cursor=at(result,floats);
    return result;
}

static int allocate_state(GPUModel *g)
{
    GptConfig c=g->c;
    size_t n=(size_t)g->max_b*g->max_t, nd=n*c.width, nl=c.layers, nv=n*c.vocab_size;
    size_t attention=n*c.heads*g->max_t;
    /* CUDA kernels use signed 32-bit indices. Check the largest views first. */
    if (n>INT_MAX || nd>INT_MAX/4 || attention>INT_MAX || nv>INT_MAX || g->count>INT_MAX)
        return 0;
    size_t total=3*n+(nl+1)*nd+nl*(15*nd+attention+4*n)+nd+2*n+nv+10*nd+attention;
    g->stride=nd;
    g->cache=calloc(nl,sizeof(*g->cache));
    g->host_loss=malloc(n*sizeof(float));
    if (!g->cache || !g->host_loss || !CUDA(alloc(&g->training,16*g->count)) ||
        !CUDA(alloc(&g->arena,4*total))) return 0;
    Dev p=g->training;
    g->weights=take(&p,g->count); g->grad=take(&p,g->count); g->m=take(&p,g->count); g->v=take(&p,g->count);
    if (!CUDA(zero(g->grad,0,3*g->count))) return 0;
    p=g->arena;
    g->input=take(&p,n); g->target=take(&p,n); g->loss=take(&p,n);
    g->hidden=take(&p,(nl+1)*nd);
    for (size_t i=0;i<nl;i++) {
        Layer *a=&g->cache[i];
        a->norm1=take(&p,nd); a->qkv=take(&p,3*nd); a->attention=take(&p,attention);
        a->mix=take(&p,nd); a->mid=take(&p,nd); a->norm2=take(&p,nd);
        a->pre=take(&p,4*nd); a->gelu=take(&p,4*nd);
        a->mean1=take(&p,n); a->inv1=take(&p,n); a->mean2=take(&p,n); a->inv2=take(&p,n);
    }
    g->final=take(&p,nd); g->mean=take(&p,n); g->inv=take(&p,n); g->logits=take(&p,nv);
    g->dx=take(&p,nd); g->dn=take(&p,nd); g->dmix=take(&p,nd);
    g->dqkv=take(&p,3*nd); g->dff=take(&p,4*nd); g->dscores=take(&p,attention);
    return p==at(g->arena,total);
}

GPUModel *gpu_create(const Gpt *src,int b,int t)
{
    if (sizeof(void*)!=8 || !src || !src->parameters || src->parameter_count==0 ||
        src->parameter_count!=gpt_parameter_count(src->config) || b<1 || t<1 ||
        t>src->config.context || (int64_t)b*t>INT_MAX || !load_api()) return NULL;
    GPUModel *g=calloc(1,sizeof(*g));
    if (!g) return NULL;
    g->c=src->config; g->count=src->parameter_count; g->steps=src->steps;
    int d=g->c.width;
    g->matrices=(size_t)(g->c.vocab_size+g->c.context)*d+12u*g->c.layers*(size_t)d*d;
    g->max_b=b; g->max_t=t;
    /* cublasComputeType_t values from CUDA 12 cublas_api.h. FAST_TF32
       enables Tensor Cores while retaining float storage and accumulation. */
    g->compute=d>=64?77:68;
    if (!CUDA(init(0)) || !CUDA(device(&g->device,0)) || !CUDA(retain(&g->context,g->device)) ||
        !CUDA(current(g->context)) || api.make_blas(&g->blas) || !compile_kernels(g) ||
        !allocate_state(g) || !CUDA(upload(g->weights,src->parameters,4*g->count))) {
        gpu_destroy(g);
        return NULL;
    }
    return g;
}

void gpu_destroy(GPUModel *g)
{
    if (!g) return;
    if (g->context) api.current(g->context);
    if (g->arena) api.dealloc(g->arena);
    if (g->training) api.dealloc(g->training);
    if (g->blas) api.destroy_blas(g->blas);
    if (g->module) api.unload(g->module);
    if (g->context) api.release(g->device);
    free(g->host_loss); free(g->cache); free(g);
}

int gpu_export(const GPUModel *g,Gpt *dest)
{
    if (!g || !dest || !dest->parameters || dest->parameter_count!=g->count ||
        dest->config.vocab_size!=g->c.vocab_size || dest->config.context!=g->c.context ||
        dest->config.width!=g->c.width || dest->config.heads!=g->c.heads ||
        dest->config.layers!=g->c.layers) return 0;
    if (!CUDA(download(dest->parameters,g->weights,4*g->count))) return 0;
    dest->steps=g->steps;
    return 1;
}

typedef struct {
    Dev qkv,proj,expand,contract,g1,b1,bq,bp,g2,b2,be,bc;
} Params;

static Params params(const GPUModel *g,Dev base,int layer)
{
    size_t d=g->c.width, square=d*d;
    Dev m=at(base,((size_t)g->c.vocab_size+g->c.context)*d+12*layer*square);
    Dev v=at(base,g->matrices+13*layer*d);
    Params p;
    p.qkv=take(&m,3*square); p.proj=take(&m,square);
    p.expand=take(&m,4*square); p.contract=take(&m,4*square);
    p.g1=take(&v,d); p.b1=take(&v,d); p.bq=take(&v,3*d); p.bp=take(&v,d);
    p.g2=take(&v,d); p.b2=take(&v,d); p.be=take(&v,4*d); p.bc=take(&v,d);
    return p;
}

static int run(GPUModel *g,enum Kernel id,unsigned blocks,void **args)
{
    return driver_ok(api.launch(g->kernel[id],blocks,1,1,128,1,1,0,NULL,args,NULL),names[id]);
}
#define RUN(id,blocks,...) run(g,id,blocks,(void*[]){__VA_ARGS__})
static unsigned groups(int n) { return 1u+((unsigned)n-1u)/128u; }

/* cuBLAS sees the transpose of each row-major buffer. Leading dimensions
   refer to the physical layout of the operands before applying op(A/B). */
static int mm(GPUModel *g,int ta,int tb,int m,int n,int k,Dev a,int lda,Dev b,int ldb,
              Dev out,int ldc,float beta)
{
    const float alpha=1;
    int status=api.gemm(g->blas,ta,tb,m,n,k,&alpha,(const void *)(uintptr_t)a,0,lda,
                        (const void *)(uintptr_t)b,0,ldb,&beta,(void *)(uintptr_t)out,0,ldc,g->compute,-1);
    if (status) fprintf(stderr,"cublasGemmEx failed (%d; %d x %d x %d)\n",status,m,n,k);
    return status==0;
}

static int dense(GPUModel *g,Dev x,Dev w,Dev bias,Dev y,int n,int in,int out)
{
    int total=n*out;
    return mm(g,1,0,out,n,in,w,in,x,in,y,out,0) &&
           (!bias || RUN(ADD,groups(total),&y,&bias,&total,&out));
}

static int dense_back(GPUModel *g,Dev x,Dev w,Dev dy,Dev dw,Dev db,Dev dx,int n,int in,int out)
{
    return mm(g,0,0,in,n,out,w,in,dy,out,dx,in,0) &&
           mm(g,0,1,in,out,n,x,in,dy,out,dw,in,1) &&
           (!db || RUN(BIAS_BACK,groups(out),&dy,&db,&n,&out));
}

static int forward(GPUModel *g,int batch,int t)
{
    int n=batch*t,d=g->c.width,v=g->c.vocab_size,h=g->c.heads;
    int nd=n*d, nf=4*nd, backward=0;
    if (!RUN(EMBED,groups(nd),&g->hidden,&g->weights,&g->input,&n,&d,&t,&v)) return 0;
    for (int i=0;i<g->c.layers;i++) {
        Layer *a=&g->cache[i];
        Params p=params(g,g->weights,i);
        Dev x=at(g->hidden,i*g->stride),y=at(x,g->stride);
        if (!RUN(NORM,n,&x,&p.g1,&p.b1,&a->norm1,&a->mean1,&a->inv1,&d) ||
            !dense(g,a->norm1,p.qkv,p.bq,a->qkv,n,d,3*d) ||
            !RUN(ATTENTION,n*h,&a->qkv,&a->attention,&a->mix,&t,&d,&h) ||
            !dense(g,a->mix,p.proj,p.bp,a->mid,n,d,d) ||
            !RUN(ADD,groups(nd),&a->mid,&x,&nd,&nd) ||
            !RUN(NORM,n,&a->mid,&p.g2,&p.b2,&a->norm2,&a->mean2,&a->inv2,&d) ||
            !dense(g,a->norm2,p.expand,p.be,a->pre,n,d,4*d) ||
            !RUN(GELU,groups(nf),&a->pre,&a->gelu,&nf,&backward) ||
            !dense(g,a->gelu,p.contract,p.bc,y,n,4*d,d) ||
            !RUN(ADD,groups(nd),&y,&a->mid,&nd,&nd)) return 0;
    }
    Dev x=at(g->hidden,g->c.layers*g->stride);
    Dev gamma=at(g->weights,g->count-2*d),beta=at(gamma,d);
    return RUN(NORM,n,&x,&gamma,&beta,&g->final,&g->mean,&g->inv,&d) &&
           dense(g,g->final,g->weights,0,g->logits,n,d,v);
}

static int backward(GPUModel *g,int batch,int t)
{
    int n=batch*t,d=g->c.width,v=g->c.vocab_size,h=g->c.heads;
    int nd=n*d,nf=4*nd,back=1;
    Dev last=at(g->hidden,g->c.layers*g->stride), gamma=at(g->weights,g->count-2*d);
    Dev dg=at(g->grad,g->count-2*d),db=at(dg,d);
    /* Cross-entropy replaced logits with dL/dlogits. The tied projection
       writes its weight gradient into the same tensor as token embeddings. */
    if (!dense_back(g,g->final,g->weights,g->logits,g->grad,0,g->dn,n,d,v) ||
        !CUDA(zero(g->dx,0,(size_t)nd)) ||
        !RUN(NORM_BACK,n,&last,&gamma,&g->dn,&g->mean,&g->inv,&dg,&db,&g->dx,&d)) return 0;
    for (int i=g->c.layers-1;i>=0;i--) {
        Layer *a=&g->cache[i];
        Params p=params(g,g->weights,i), q=params(g,g->grad,i);
        Dev x=at(g->hidden,i*g->stride);
        if (!dense_back(g,a->gelu,p.contract,g->dx,q.contract,q.bc,g->dff,n,4*d,d) ||
            !RUN(GELU,groups(nf),&a->pre,&g->dff,&nf,&back) ||
            !dense_back(g,a->norm2,p.expand,g->dff,q.expand,q.be,g->dn,n,d,4*d) ||
            !RUN(NORM_BACK,n,&a->mid,&p.g2,&g->dn,&a->mean2,&a->inv2,&q.g2,&q.b2,&g->dx,&d) ||
            !dense_back(g,a->mix,p.proj,g->dx,q.proj,q.bp,g->dmix,n,d,d) ||
            !RUN(SCORE_BACK,n*h,&a->qkv,&a->attention,&a->mix,&g->dmix,&g->dscores,&t,&d,&h) ||
            !RUN(QKV_BACK,groups(nd),&a->qkv,&a->attention,&g->dmix,&g->dscores,&g->dqkv,&n,&t,&d,&h) ||
            !dense_back(g,a->norm1,p.qkv,g->dqkv,q.qkv,q.bq,g->dn,n,d,3*d) ||
            !RUN(NORM_BACK,n,&x,&p.g1,&g->dn,&a->mean1,&a->inv1,&q.g1,&q.b1,&g->dx,&d)) return 0;
    }
    return RUN(EMBED_BACK,groups(nd),&g->grad,&g->dx,&g->input,&n,&d,&t,&v);
}

double gpu_pass(GPUModel *g,const int *input,const int *target,int batch,int t,int differentiate)
{
    if (!g || !input || batch<1 || batch>g->max_b || t<1 || t>g->max_t ||
        (differentiate!=0 && differentiate!=1) || (differentiate && !target)) return NAN;
    int n=batch*t,v=g->c.vocab_size;
    g->has_logits=0;
    for(int i=0;i<n;i++)
        if(input[i]<0 || input[i]>=v || (target && (target[i]<0 || target[i]>=v))) return NAN;
    if (!CUDA(upload(g->input,input,sizeof(int)*(size_t)n)) || !forward(g,batch,t)) return NAN;
    g->last_b=batch; g->last_t=t;
    if (!target) { g->has_logits=1; return 0; }
    if (!CUDA(upload(g->target,target,sizeof(int)*(size_t)n)) ||
        !RUN(XENT,n,&g->logits,&g->target,&g->loss,&v,&differentiate) ||
        !CUDA(download(g->host_loss,g->loss,sizeof(float)*(size_t)n))) return NAN;
    double total=0;
    for(int i=0;i<n;i++) total+=g->host_loss[i];
    if (!isfinite(total)) return NAN;
    if (differentiate) {
        if (UINT64_MAX-g->tokens<(uint64_t)n || !backward(g,batch,t)) return NAN;
        g->tokens+=(uint64_t)n;
    }
    return total/n;
}

int gpu_last_logits(GPUModel *g,int b,int t,float *out)
{
    if (!g || !g->has_logits || !out || b<0 || b>=g->last_b || t<0 || t>=g->last_t) return 0;
    size_t idx=((size_t)b*g->last_t+t)*g->c.vocab_size;
    return CUDA(download(out,at(g->logits,idx),g->c.vocab_size*sizeof(float)));
}

int gpu_update(GPUModel *g,double lr,double wd,double clip,double *norm_out)
{
    if (!g || !g->tokens || g->steps==UINT64_MAX || !isfinite(lr) || lr<=0 || lr>1 ||
        !isfinite(wd) || wd<0 || wd>1 || !isfinite(clip) || clip<0) return 0;
    int n=(int)g->count,mat=(int)g->matrices;
    float sum_norm=0;
    if (api.nrm2(g->blas,n,(const float *)(uintptr_t)g->grad,1,&sum_norm) || !isfinite(sum_norm)) return 0;
    double norm=sum_norm/(double)g->tokens;
    float scale=(float)((clip>0 && norm>clip?clip/norm:1)/(double)g->tokens);
    float rate=(float)lr,decay=(float)wd;
    double step=(double)(g->steps+1);
    float c1=(float)(1/(1-pow(0.9,step))),c2=(float)(1/(1-pow(0.999,step)));
    if (!RUN(ADAM,groups(n),&g->weights,&g->grad,&g->m,&g->v,&n,&mat,&scale,&rate,&decay,&c1,&c2)) return 0;
    ++g->steps; g->tokens=0; g->has_logits=0;
    if(norm_out) *norm_out=norm;
    return 1;
}

static int parity(GptConfig config)
{
    enum { B=2,T=7 };
    Gpt cpu={0},copy={0};
    GPUModel *g=NULL;
    int x[B*T],y[B*T],ok=0;
    float *logits=NULL,*gradient=NULL;
    double max_logit=0,max_relative_gradient=0,max_update_error=0;
    if (!gpt_init(&cpu,config,81) || !gpt_init(&copy,config,81) ||
        !(g=gpu_create(&cpu,B,T))) goto done;
    logits=malloc((size_t)config.vocab_size*sizeof(float));
    gradient=malloc(cpu.parameter_count*sizeof(float));
    if (!logits || !gradient) goto done;
    for (int step=0;step<2;step++) {
        for (int i=0;i<B*T;i++) {
            x[i]=(3*i+7*step)%config.vocab_size;
            y[i]=(2*i+1+step)%config.vocab_size;
        }
        if (gpu_pass(g,x,NULL,B,T,0)!=0) goto done;
        double ref_loss=0;
        for (int b=0;b<B;b++) {
            if (!gpt_forward(&cpu,x+b*T,T)) goto done;
            for(int j=0;j<T;j++) {
                if (!gpu_last_logits(g,b,j,logits)) goto done;
                const float *expected=gpt_logits(&cpu,j);
                for(int k=0;k<config.vocab_size;k++)
                    max_logit=fmax(max_logit,fabs((double)logits[k]-expected[k]));
            }
            ref_loss+=gpt_backward(&cpu,y+b*T)/B;
        }
        double loss=gpu_pass(g,x,y,B,T,1);
        if (!isfinite(loss) || fabs(loss-ref_loss)>0.002) goto done;
        /* A shorter second microbatch exercises accumulation and non-capacity strides. */
        if (!gpt_forward(&cpu,x,3)) goto done;
        ref_loss=gpt_backward(&cpu,y);
        loss=gpu_pass(g,x,y,1,3,1);
        if (!isfinite(loss) || fabs(loss-ref_loss)>0.002 ||
            !CUDA(download(gradient,g->grad,sizeof(float)*g->count))) goto done;
        const float *ref=gpt_gradient(&cpu);
        double sq=0,err=0;
        for(size_t i=0;i<g->count;i++) {
            double diff=(double)gradient[i]-ref[i];
            err+=diff*diff; sq+=(double)ref[i]*ref[i];
        }
        max_relative_gradient=fmax(max_relative_gradient,sqrt(err/sq));
        memcpy(gradient,cpu.parameters,g->count*sizeof(float));
        double gpu_norm,cpu_norm;
        if (!gpu_update(g,.001,.03,.25,&gpu_norm) || !gpt_step(&cpu,.001,.03,.25,&cpu_norm) ||
            !gpu_export(g,&copy) || fabs(gpu_norm-cpu_norm)>.002*cpu_norm) goto done;
        sq=0; err=0;
        for(size_t i=0;i<g->count;i++) {
            double diff=(double)copy.parameters[i]-cpu.parameters[i];
            double applied=(double)cpu.parameters[i]-gradient[i];
            err+=diff*diff; sq+=applied*applied;
        }
        max_update_error=fmax(max_update_error,sqrt(err/sq));
    }
    printf("CUDA parity d=%d B=2 T=7 | max logit delta %.3g | gradient relative L2 %.3g "
           "| AdamW delta relative L2 %.3g\n",config.width,max_logit,max_relative_gradient,max_update_error);
    ok=max_logit<.003 && max_relative_gradient<.004 && max_update_error<.08;
done:
    gpu_destroy(g); gpt_free(&cpu); gpt_free(&copy); free(logits); free(gradient);
    return ok;
}

int gpu_selftest(void)
{
    return parity((GptConfig){23,9,12,3,2}) && parity((GptConfig){73,9,64,4,2});
}
