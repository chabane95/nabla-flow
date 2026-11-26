/* Compiled at runtime by NVRTC. Dense products use cuBLAS on the same stream.
   Every reduction below uses a block of 128 threads. */
static __device__ float reduce_sum(float v, float *s)
{
    int i = threadIdx.x;
    s[i] = v;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (i < stride) s[i] += s[i + stride];
        __syncthreads();
    }
    v = s[0];
    __syncthreads();
    return v;
}

static __device__ float reduce_max(float v, float *s)
{
    int i = threadIdx.x;
    s[i] = v;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (i < stride) s[i] = fmaxf(s[i],s[i+stride]);
        __syncthreads();
    }
    v = s[0];
    __syncthreads();
    return v;
}

extern "C" __global__ void embedding(float *out, const float *w, const int *ids,
                                     int n, int d, int t, int vocab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n*d) return;
    int row = i/d, j = i%d;
    out[i] = w[ids[row]*d+j] + w[(vocab+row%t)*d+j];
}

extern "C" __global__ void embedding_back(float *dw, const float *dx, const int *ids,
                                          int n, int d, int t, int vocab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n*d) return;
    int row = i/d, j = i%d;
    atomicAdd(&dw[ids[row]*d+j],dx[i]);
    atomicAdd(&dw[(vocab+row%t)*d+j],dx[i]);
}

extern "C" __global__ void layernorm(const float *x, const float *g, const float *b,
                                     float *out, float *mean, float *inv, int d)
{
    __shared__ float shared[128];
    int row = blockIdx.x, i = threadIdx.x;
    x += row*d; out += row*d;
    float v = 0;
    for (int j=i; j<d; j+=128) v += x[j];
    float m = reduce_sum(v,shared)/d;
    v = 0;
    for (int j=i; j<d; j+=128) { float z = x[j]-m; v += z*z; }
    float r = rsqrtf(reduce_sum(v,shared)/d + 1e-5f);
    if (i==0) { mean[row]=m; inv[row]=r; }
    for (int j=i; j<d; j+=128) out[j] = (x[j]-m)*r*g[j]+b[j];
}

extern "C" __global__ void layernorm_back(const float *x, const float *g,
    const float *dy, const float *mean, const float *inv, float *dg, float *db, float *dx, int d)
{
    __shared__ float shared[128];
    int row = blockIdx.x, i = threadIdx.x;
    x += row*d; dx += row*d; dy += row*d;
    float m = mean[row], r=inv[row], s=0, p=0;
    for (int j=i;j<d;j+=128) {
        float z=(x[j]-m)*r, dz=dy[j]*g[j];
        s+=dz; p+=dz*z; atomicAdd(dg+j,dy[j]*z); atomicAdd(db+j,dy[j]);
    }
    s=reduce_sum(s,shared)/d; p=reduce_sum(p,shared)/d;
    for (int j=i;j<d;j+=128) dx[j] += r*(dy[j]*g[j]-s-((x[j]-m)*r)*p);
}

extern "C" __global__ void add_vector(float *x,const float *b,int n,int dim)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) x[i] += b[i%dim];
}

extern "C" __global__ void bias_back(const float *dy,float *db,int n,int d)
{
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if (j>=d) return;
    float sum=0;
    for (int i=0;i<n;i++) sum+=dy[i*d+j];
    db[j]+=sum;
}

extern "C" __global__ void gelu_kernel(const float *x,float *out,int n,int backward)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float a=x[i], th=tanhf(0.7978845608f*a*(1.0f+0.044715f*a*a));
    if (backward)
        out[i] *= 0.5f*(1+th)+0.5f*a*(1-th*th)*0.7978845608f*(1+0.134145f*a*a);
    else out[i]=0.5f*a*(1+th);
}

/* One block owns one (batch, query position, head) row. Tokens in separate
   sequences have disjoint addresses, even when they share the minibatch. */
extern "C" __global__ void attention(const float *qkv,float *weights,float *mix,int t,int d,int heads)
{
    __shared__ float shared[128];
    int h=blockIdx.x%heads, row=blockIdx.x/heads, q=row%t, base=(row/t)*t, kdim=d/heads;
    int lane=threadIdx.x;
    weights += ((base/t*heads+h)*t+q)*t;
    const float *query=qkv+row*3*d+h*kdim;
    float high=-3.40282347e38f;
    for (int j=lane;j<=q;j+=128) {
        const float *key=qkv+(base+j)*3*d+d+h*kdim;
        float v=0;
        for(int k=0;k<kdim;k++) v+=query[k]*key[k];
        weights[j]=v*rsqrtf((float)kdim); high=fmaxf(high,weights[j]);
    }
    high=reduce_max(high,shared);
    float sum=0;
    for(int j=lane;j<=q;j+=128){weights[j]=expf(weights[j]-high);sum+=weights[j];}
    sum=reduce_sum(sum,shared);
    for(int j=lane;j<t;j+=128) weights[j]=j<=q?weights[j]/sum:0;
    __syncthreads();
    for(int k=lane;k<kdim;k+=128){
        float y=0;
        for(int j=0;j<=q;j++) y+=weights[j]*qkv[(base+j)*3*d+2*d+h*kdim+k];
        mix[row*d+h*kdim+k]=y;
    }
}

extern "C" __global__ void attention_scores_back(const float *qkv,const float *w,const float *mix,
    const float *dy,float *scores,int t,int d,int heads)
{
    __shared__ float shared[128];
    int h=blockIdx.x%heads,row=blockIdx.x/heads,q=row%t,base=(row/t)*t,kdim=d/heads;
    int offset=((base/t*heads+h)*t+q)*t;
    int lane=threadIdx.x;
    dy+=row*d+h*kdim; mix+=row*d+h*kdim;
    float expect=0;
    for(int k=lane;k<kdim;k+=128) expect+=dy[k]*mix[k];
    expect=reduce_sum(expect,shared);
    for(int j=lane;j<=q;j+=128) {
        float v=0;
        for(int k=0;k<kdim;k++) v+=dy[k]*qkv[(base+j)*3*d+2*d+h*kdim+k];
        scores[offset+j]=w[offset+j]*(v-expect)*rsqrtf((float)kdim);
    }
}

extern "C" __global__ void attention_qkv_back(const float *qkv,const float *w,const float *dy,
    const float *scores,float *dqkv,int n,int t,int d,int heads)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*d) return;
    int row=i/d,col=i%d,q=row%t,base=(row/t)*t,h=col/(d/heads);
    int at=(base/t*heads+h)*t*t;
    float dq=0,dk=0,dv=0;
    for(int j=0;j<=q;j++) dq+=scores[at+q*t+j]*qkv[(base+j)*3*d+d+col];
    for(int j=q;j<t;j++){
        dk+=scores[at+j*t+q]*qkv[(base+j)*3*d+col];
        dv+=w[at+j*t+q]*dy[(base+j)*d+col];
    }
    dqkv[row*3*d+col]=dq; dqkv[row*3*d+d+col]=dk; dqkv[row*3*d+2*d+col]=dv;
}

extern "C" __global__ void cross_entropy(float *logits,const int *target,float *loss,int v,int backward)
{
    __shared__ float shared[128];
    int row=blockIdx.x, lane=threadIdx.x; logits+=(size_t)row*v;
    float high=-3.40282347e38f;
    for(int k=lane;k<v;k+=128) high=fmaxf(high,logits[k]);
    high=reduce_max(high,shared);
    float sum=0;
    for(int k=lane;k<v;k+=128) sum+=expf(logits[k]-high);
    sum=reduce_sum(sum,shared);
    int truth=target[row];
    if(!lane) loss[row]=(high-logits[truth])+logf(sum);
    __syncthreads();
    if(backward)
        for(int k=lane;k<v;k+=128) logits[k]=expf(logits[k]-high)/sum-(k==truth);
}

extern "C" __global__ void adamw(float *w,float *grad,float *m,float *v,
    int n,int matrices,float scale,float lr,float wd,float c1,float c2)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float g=grad[i]*scale;
    float first=.9f*m[i]+.1f*g, second=.999f*v[i]+.001f*g*g;
    w[i] -= lr*(first*c1/(sqrtf(second*c2)+1e-8f)+(i<matrices?wd*w[i]:0));
    m[i]=first; v[i]=second; grad[i]=0;
}
