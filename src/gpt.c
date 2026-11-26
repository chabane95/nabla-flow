#include "gpt.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Matrices have one contiguous row per output. All matrices precede the
   normalization scales, offsets, and biases in the checkpoint. */
typedef struct {
    float *qkv, *project, *expand, *contract;
    float *gamma1, *beta1, *bqkv, *bproject;
    float *gamma2, *beta2, *bexpand, *bcontract;
} Weights;

typedef struct {
    float *n1, *qkv, *attention, *mixed, *mid, *n2, *pre, *gelu;
    float *mean1, *inv1, *mean2, *inv2;
    int capacity;
} Cache;

typedef struct GptState {
    size_t matrices;
    uint64_t accumulated;
    int length, capacity;
    int *tokens;
    float *arena, *grad, *m, *v, *proposal;
    float *hidden, *final, *mean, *inv, *logits;
    float *dx, *dn, *dmixed, *dqkv, *dff, *dlogit;
    Cache *cache;
} State;

static float *take(float **cursor, size_t n)
{
    float *p = *cursor;
    *cursor += n;
    return p;
}

static Weights layer_weights(const Gpt *model, float *base, int layer)
{
    const GptConfig c = model->config;
    size_t d = (size_t)c.width, square = d*d;
    float *mat = base + ((size_t)c.vocab_size + c.context)*d + 12*layer*square;
    float *vec = base + model->state->matrices + 13*layer*d;
    Weights w;
    w.qkv = take(&mat, 3*square);
    w.project = take(&mat, square);
    w.expand = take(&mat, 4*square);
    w.contract = take(&mat, 4*square);
    w.gamma1 = take(&vec, d);
    w.beta1 = take(&vec, d);
    w.bqkv = take(&vec, 3*d);
    w.bproject = take(&vec, d);
    w.gamma2 = take(&vec, d);
    w.beta2 = take(&vec, d);
    w.bexpand = take(&vec, 4*d);
    w.bcontract = take(&vec, d);
    return w;
}

static float *final_gamma(const Gpt *model, float *base)
{
    return base + model->parameter_count - 2*model->config.width;
}

/* SplitMix64, then Box-Muller. Open interval excludes log(0). */
static double uniform(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return ((double)(z >> 12) + 0.5) * (1.0 / 4503599627370496.0);
}

static void initialize(Gpt *model, uint32_t seed)
{
    uint64_t rng = seed;
    size_t n = model->state->matrices;
    for (size_t i = 0; i < n; i += 2) {
        double radius = 0.02 * sqrt(-2.0 * log(uniform(&rng)));
        double angle = 6.2831853071795864769 * uniform(&rng);
        model->parameters[i] = (float)(radius * cos(angle));
        if (i+1 < n)
            model->parameters[i+1] = (float)(radius * sin(angle));
    }
    int d = model->config.width;
    float residual_scale = 1.0f / sqrtf(2.0f * model->config.layers);
    for (int l = 0; l < model->config.layers; ++l) {
        Weights w = layer_weights(model, model->parameters, l);
        for (int i = 0; i < d*d; ++i) w.project[i] *= residual_scale;
        for (int i = 0; i < 4*d*d; ++i) w.contract[i] *= residual_scale;
        for (int j = 0; j < d; ++j) w.gamma1[j] = w.gamma2[j] = 1.0f;
    }
    float *gamma = final_gamma(model, model->parameters);
    for (int j = 0; j < d; ++j) gamma[j] = 1.0f;
}

size_t gpt_parameter_count(GptConfig c)
{
    if (c.vocab_size < 2 || c.vocab_size > 65536 ||
        c.context < 1 || c.context > 2048 || c.width < 2 || c.width > 768 ||
        c.heads < 1 || c.heads > c.width || c.width % c.heads ||
        c.layers < 1 || c.layers > 24)
        return 0;
    uint64_t t = (uint64_t)c.context, d = (uint64_t)c.width, v = (uint64_t)c.vocab_size;
    uint64_t l = (uint64_t)c.layers;
    uint64_t mat = (v+t)*d + 12*l*d*d;
    uint64_t n = mat + 13*l*d + 2*d;
    return n <= SIZE_MAX/sizeof(float) ? (size_t)n : 0;
}

int gpt_init(Gpt *model, GptConfig c, uint32_t seed)
{
    size_t n = gpt_parameter_count(c);
    if (!model || model->state || !n) return 0;
    State *s = calloc(1, sizeof(*s));
    if (!s)
        return 0;
    model->state = s;
    model->parameters = calloc(n, sizeof(float));
    s->tokens = calloc((size_t)c.context, sizeof(int));
    s->cache = calloc((size_t)c.layers, sizeof(Cache));
    if (!model->parameters || !s->tokens || !s->cache) {
        gpt_free(model);
        return 0;
    }
    model->config = c;
    model->steps = 0;
    model->parameter_count = (size_t)n;
    s->matrices = n - ((size_t)13*c.layers+2)*c.width;
    initialize(model, seed);
    return 1;
}

static int reserve_window(Gpt *model, int length)
{
    GptConfig c = model->config;
    State *s = model->state;
    if (length <= s->capacity) return 1;
    int cap = 2*s->capacity;
    if (cap < length) cap = length;
    if (cap > c.context) cap = c.context;
    uint64_t t = (uint64_t)cap, d = (uint64_t)c.width, v = (uint64_t)c.vocab_size;
    uint64_t l = (uint64_t)c.layers, h = (uint64_t)c.heads, td = t*d;
    uint64_t floats = (l+1)*td + l*(15*td+h*t*t+4*t) + td + 2*t + t*v + 10*td + v;
    if (floats > SIZE_MAX/sizeof(float) || floats > UINT64_C(4294967296)/sizeof(float))
        return 0;
    /* Forward overwrites all live rows. Releasing old activations first avoids
       doubling peak memory when growing a large inspection window. */
    free(s->arena);
    s->arena = malloc((size_t)floats * sizeof(float));
    s->capacity = 0;
    if (!s->arena) return 0;
    s->capacity = cap;
    float *p = s->arena;
    s->hidden = take(&p, (size_t)((l+1)*td));
    for (int i = 0; i < c.layers; ++i) {
        Cache *a = &s->cache[i];
        a->capacity = cap;
        a->n1 = take(&p, (size_t)td);
        a->qkv = take(&p, (size_t)(3*td));
        a->attention = take(&p, (size_t)(h*t*t));
        a->mixed = take(&p, (size_t)td);
        a->mid = take(&p, (size_t)td);
        a->n2 = take(&p, (size_t)td);
        a->pre = take(&p, (size_t)(4*td));
        a->gelu = take(&p, (size_t)(4*td));
        a->mean1 = take(&p, (size_t)t); a->inv1 = take(&p, (size_t)t);
        a->mean2 = take(&p, (size_t)t); a->inv2 = take(&p, (size_t)t);
    }
    s->final = take(&p, (size_t)td);
    s->mean = take(&p, (size_t)t); s->inv = take(&p, (size_t)t);
    s->logits = take(&p, (size_t)(t*v));
    s->dx = take(&p, (size_t)td); s->dn = take(&p, (size_t)td);
    s->dmixed = take(&p, (size_t)td); s->dqkv = take(&p, (size_t)(3*td));
    s->dff = take(&p, (size_t)(4*td)); s->dlogit = take(&p, (size_t)v);
    return 1;
}

void gpt_free(Gpt *model)
{
    if (!model) return;
    if (model->state) {
        free(model->parameters);
        free(model->state->grad);
        free(model->state->m);
        free(model->state->tokens);
        free(model->state->cache);
        free(model->state->arena);
        free(model->state);
    }
    *model = (Gpt){0};
}

static float dot(const float *a, const float *b, int n)
{
    float result = 0.0f;
    for (int i = 0; i < n; ++i) result += a[i] * b[i];
    return result;
}

static void dense(const float *x, const float *w, const float *b, float *out,
                  int length, int inputs, int outputs)
{
    for (int t = 0; t < length; ++t) {
        for (int j = 0; j < outputs; ++j)
            out[j] = dot(x, w + (size_t)j*inputs, inputs) + (b ? b[j] : 0.0f);
        x += inputs;
        out += outputs;
    }
}

static void normalize(const float *x, const float *gamma, const float *beta,
                      float *out, float *means, float *invs, int length, int d)
{
    for (int t = 0; t < length; ++t, x += d, out += d) {
        double sum = 0.0, sq = 0.0;
        for (int j = 0; j < d; ++j) sum += x[j];
        float mean = (float)(sum / d);
        for (int j = 0; j < d; ++j) {
            double z = (double)x[j] - mean;
            sq += z*z;
        }
        means[t] = mean;
        invs[t] = (float)(1.0 / sqrt(sq/d + 1e-5));
        for (int j = 0; j < d; ++j)
            out[j] = (x[j] - mean) * invs[t] * gamma[j] + beta[j];
    }
}

/* tanh approximation of GELU, using the same formula in both passes. */
static float gelu_tanh(float x)
{
    return tanhf(0.7978845608f * x * (1.0f + 0.044715f*x*x));
}

static float *attention_row(Cache *a, int head, int t)
{
    return a->attention + ((size_t)head*a->capacity + t)*a->capacity;
}

static void attention_forward(Cache *a, GptConfig c, int length)
{
    int d = c.width, kdim = d/c.heads;
    float scale = 1.0f / sqrtf((float)kdim);
    memset(a->mixed, 0, (size_t)length*d*sizeof(float));
    for (int t = 0; t < length; ++t) {
        for (int h = 0; h < c.heads; ++h) {
            const float *q = a->qkv + t*3*d + h*kdim;
            float *row = attention_row(a, h, t);
            float largest = -INFINITY, sum = 0.0f;
            for (int j = 0; j <= t; ++j) {
                const float *key = a->qkv + j*3*d + d + h*kdim;
                row[j] = dot(q, key, kdim) * scale;
                if (row[j] > largest) largest = row[j];
            }
            for (int j = 0; j <= t; ++j) {
                row[j] = expf(row[j] - largest);
                sum += row[j];
            }
            float *mixed = a->mixed + t*d + h*kdim;
            for (int j = 0; j <= t; ++j) {
                const float *value = a->qkv + j*3*d + 2*d + h*kdim;
                row[j] /= sum;
                for (int i = 0; i < kdim; ++i) mixed[i] += row[j] * value[i];
            }
        }
    }
}

int gpt_forward(Gpt *model, const int *tokens, int length)
{
    if (!model || !model->state) return 0;
    GptConfig c = model->config;
    State *s = model->state;
    s->length = 0;
    if (!tokens || length < 1 || length > c.context || !reserve_window(model, length)) return 0;
    int d = c.width;
    float *pos = model->parameters + (size_t)c.vocab_size*d;
    for (int t = 0; t < length; ++t) {
        if (tokens[t] < 0 || tokens[t] >= c.vocab_size) return 0;
        s->tokens[t] = tokens[t];
        const float *token = model->parameters + (size_t)tokens[t]*d;
        for (int j = 0; j < d; ++j) s->hidden[t*d+j] = token[j] + pos[t*d+j];
    }
    size_t stride = (size_t)s->capacity*d;
    float *x = s->hidden;
    for (int l = 0; l < c.layers; ++l, x += stride) {
        Weights w = layer_weights(model, model->parameters, l);
        Cache *a = &s->cache[l];
        normalize(x, w.gamma1, w.beta1, a->n1, a->mean1, a->inv1, length, d);
        dense(a->n1, w.qkv, w.bqkv, a->qkv, length, d, 3*d);
        attention_forward(a, c, length);
        dense(a->mixed, w.project, w.bproject, a->mid, length, d, d);
        for (int i = 0; i < length*d; ++i) a->mid[i] += x[i];
        normalize(a->mid, w.gamma2, w.beta2, a->n2, a->mean2, a->inv2, length, d);
        dense(a->n2, w.expand, w.bexpand, a->pre, length, d, 4*d);
        for (int i = 0; i < length*4*d; ++i)
            a->gelu[i] = 0.5f * a->pre[i] * (1.0f + gelu_tanh(a->pre[i]));
        dense(a->gelu, w.contract, w.bcontract, x+stride, length, 4*d, d);
        for (int i = 0; i < length*d; ++i) x[stride+i] += a->mid[i];
    }
    float *gamma = final_gamma(model, model->parameters);
    normalize(x, gamma, gamma+d, s->final, s->mean, s->inv, length, d);
    dense(s->final, model->parameters, NULL, s->logits, length, d, c.vocab_size);
    for (size_t i = 0; i < (size_t)length*c.vocab_size; ++i)
        if (!isfinite(s->logits[i])) return 0;
    s->length = length;
    return 1;
}

const float *gpt_logits(const Gpt *model, int position)
{
    if (!model || !model->state || position < 0 || position >= model->state->length)
        return NULL;
    return model->state->logits + (size_t)position*model->config.vocab_size;
}

int gpt_attention_row(const Gpt *model, int layer, int head, int position, float *out)
{
    if (!gpt_logits(model, position) || !out || layer < 0 || layer >= model->config.layers ||
        head < 0 || head >= model->config.heads)
        return 0;
    Cache *a = &model->state->cache[layer];
    const float *src = attention_row(a, head, position);
    memcpy(out, src, (size_t)(position+1)*sizeof(float));
    memset(out+position+1, 0, (size_t)(model->state->length-position-1)*sizeof(float));
    return 1;
}

/* dW and db add to existing accumulated gradients. dx is scratch and is
   overwritten, allowing the residual branch to be held in a separate buffer. */
static void dense_backward(const float *x, const float *w, const float *dy,
                           float *dw, float *db, float *dx,
                           int length, int inputs, int outputs)
{
    for (int t = 0; t < length; ++t, x += inputs, dy += outputs, dx += inputs) {
        memset(dx, 0, (size_t)inputs*sizeof(float));
        for (int j = 0; j < outputs; ++j) {
            float u = dy[j];
            const float *row = w + (size_t)j*inputs;
            float *grow = dw + (size_t)j*inputs;
            if (db) db[j] += u;
            for (int i = 0; i < inputs; ++i) {
                dx[i] += row[i] * u;
                grow[i] += x[i] * u;
            }
        }
    }
}

static void normalize_backward(const float *x, const float *gamma, const float *dy,
                               const float *mean, const float *inv,
                               float *dg, float *db, float *dx, int length, int d)
{
    for (int t = 0; t < length; ++t, x += d, dy += d, dx += d) {
        double sum = 0.0, projection = 0.0;
        for (int i = 0; i < d; ++i) {
            float z = (x[i] - mean[t]) * inv[t];
            float dz = dy[i] * gamma[i];
            dg[i] += dy[i] * z;
            db[i] += dy[i];
            sum += dz;
            projection += dz * z;
        }
        float avg = (float)(sum/d), projected = (float)(projection/d);
        for (int i = 0; i < d; ++i) {
            float z = (x[i] - mean[t]) * inv[t];
            dx[i] += inv[t] * (dy[i]*gamma[i] - avg - z*projected);
        }
    }
}

static void attention_backward(Cache *a, GptConfig c, int length,
                               const float *dy, float *dqkv)
{
    int d = c.width, kdim = d/c.heads;
    float scale = 1.0f / sqrtf((float)kdim);
    memset(dqkv, 0, (size_t)length*3*d*sizeof(float));
    for (int t = 0; t < length; ++t) {
        for (int h = 0; h < c.heads; ++h) {
            int offset = h*kdim;
            const float *q = a->qkv + t*3*d + offset;
            float *dq = dqkv + t*3*d + offset;
            const float *row = attention_row(a, h, t);
            const float *dz = dy + t*d + offset;
            /* sum_j p_j <dz, V_j> = <dz, sum_j p_j V_j>. */
            float expected = dot(dz, a->mixed + t*d + offset, kdim);
            for (int j = 0; j <= t; ++j) {
                const float *key = a->qkv + j*3*d + d + offset;
                const float *value = key + d;
                float *dk = dqkv + j*3*d + d + offset;
                float *dv = dk + d;
                float ds = row[j] * (dot(dz, value, kdim) - expected) * scale;
                for (int i = 0; i < kdim; ++i) {
                    dq[i] += ds * key[i];
                    dk[i] += ds * q[i];
                    dv[i] += row[j] * dz[i];
                }
            }
        }
    }
}

/* The loss uses the shifted logits directly; target probabilities need never
   be logged and may safely underflow to zero. */
static double token_objective(Gpt *model, const int *targets, int differentiate)
{
    if (!model || !model->state || !targets || model->state->length < 1) return NAN;
    State *s = model->state;
    const int v = model->config.vocab_size, d = model->config.width;
    for (int t = 0; t < s->length; ++t)
        if (targets[t] < 0 || targets[t] >= v) return NAN;
    double loss = 0.0;
    for (int t = 0; t < s->length; ++t) {
        const float *logits = s->logits + (size_t)t*v;
        float max = logits[0];
        for (int j = 1; j < v; ++j) if (logits[j] > max) max = logits[j];
        double sum = 0.0;
        for (int j = 0; j < v; ++j) {
            double e = exp((double)logits[j] - max);
            if (differentiate) s->dlogit[j] = (float)e;
            sum += e;
        }
        loss += ((double)max - logits[targets[t]]) + log(sum);
        if (differentiate) {
            float inverse = (float)(1.0/sum);
            for (int j = 0; j < v; ++j) s->dlogit[j] *= inverse;
            s->dlogit[targets[t]] -= 1.0f;
            dense_backward(s->final+t*d, model->parameters, s->dlogit, s->grad,
                           NULL, s->dn+t*d, 1, d, v);
        }
    }
    return loss/s->length;
}

double gpt_loss(Gpt *model, const int *targets)
{
    return token_objective(model, targets, 0);
}

double gpt_backward(Gpt *model, const int *targets)
{
    if (!model || !model->state || model->state->length < 1) return NAN;
    GptConfig c = model->config;
    State *s = model->state;
    int length = s->length, d = c.width;
    if ((uint64_t)length > UINT64_MAX - s->accumulated) return NAN;
    if (!s->grad) {
        s->grad = calloc(model->parameter_count, sizeof(float));
        if (!s->grad) return NAN;
    }
    double loss = token_objective(model, targets, 1);
    if (!isfinite(loss)) return NAN;
    size_t stride = (size_t)s->capacity*d;
    float *gamma = final_gamma(model, model->parameters);
    float *dg = final_gamma(model, s->grad);
    memset(s->dx, 0, (size_t)length*d*sizeof(float));
    normalize_backward(s->hidden + c.layers*stride, gamma, s->dn, s->mean, s->inv,
                       dg, dg+d, s->dx, length, d);
    for (int l = c.layers-1; l >= 0; --l) {
        Weights w = layer_weights(model, model->parameters, l);
        Weights g = layer_weights(model, s->grad, l);
        Cache *a = &s->cache[l];
        dense_backward(a->gelu, w.contract, s->dx, g.contract, g.bcontract,
                       s->dff, length, 4*d, d);
        for (int i = 0; i < length*4*d; ++i) {
            float x = a->pre[i], th = gelu_tanh(x);
            float slope = 0.5f * (1.0f+th);
            if (fabsf(th) < 1.0f)
                slope += 0.5f*x*(1.0f-th*th)*0.7978845608f*(1.0f+0.134145f*x*x);
            s->dff[i] *= slope;
        }
        dense_backward(a->n2, w.expand, s->dff, g.expand, g.bexpand, s->dn,
                       length, d, 4*d);
        normalize_backward(a->mid, w.gamma2, s->dn, a->mean2, a->inv2,
                           g.gamma2, g.beta2, s->dx, length, d);
        dense_backward(a->mixed, w.project, s->dx, g.project, g.bproject,
                       s->dmixed, length, d, d);
        attention_backward(a, c, length, s->dmixed, s->dqkv);
        dense_backward(a->n1, w.qkv, s->dqkv, g.qkv, g.bqkv, s->dn,
                       length, d, 3*d);
        normalize_backward(s->hidden+l*stride, w.gamma1, s->dn, a->mean1, a->inv1,
                           g.gamma1, g.beta1, s->dx, length, d);
    }
    float *dpos = s->grad + (size_t)c.vocab_size*d;
    for (int t = 0; t < length; ++t) {
        float *dtoken = s->grad + (size_t)s->tokens[t]*d;
        for (int i = 0; i < d; ++i) {
            dtoken[i] += s->dx[t*d+i];
            dpos[t*d+i] += s->dx[t*d+i];
        }
    }
    s->accumulated += (uint64_t)length;
    return loss;
}

const float *gpt_gradient(const Gpt *model)
{
    return model && model->state ? model->state->grad : NULL;
}

void gpt_zero_grad(Gpt *model)
{
    if (model && model->state) {
        if (model->state->grad)
            memset(model->state->grad, 0, model->parameter_count*sizeof(float));
        model->state->accumulated = 0;
    }
}

int gpt_step(Gpt *model, double lr, double weight_decay, double clip_norm, double *grad_norm)
{
    if (!model || !model->state || model->steps == UINT64_MAX ||
        !isfinite(lr) || lr <= 0.0 || !isfinite(weight_decay) || weight_decay < 0.0 ||
        !isfinite(clip_norm) || clip_norm < 0.0)
        return 0;
    State *s = model->state;
    if (!s->accumulated) return 0;
    size_t n = model->parameter_count;
    if (!s->m) {
        if (n > SIZE_MAX/(3*sizeof(float))) return 0;
        s->m = calloc(3*n, sizeof(float));
        if (!s->m) return 0;
        s->v = s->m + n;
        s->proposal = s->v + n;
    }
    double squares = 0.0, scale = 1.0/(double)s->accumulated;
    for (size_t i = 0; i < n; ++i) squares += (double)s->grad[i]*s->grad[i];
    double norm = sqrt(squares) * scale;
    if (!isfinite(norm)) return 0;
    if (clip_norm > 0.0 && norm > clip_norm) scale *= clip_norm/norm;
    double step = (double)(model->steps+1);
    double corr1 = 1.0 / (1.0-pow(0.9, step));
    double corr2 = 1.0 / (1.0-pow(0.999, step));
    for (size_t i = 0; i < n; ++i) {
        double g = s->grad[i]*scale;
        double m = 0.9*s->m[i] + 0.1*g;
        double v = 0.999*s->v[i] + 0.001*g*g;
        double decay = i < s->matrices ? weight_decay*model->parameters[i] : 0.0;
        double update = m*corr1/(sqrt(v*corr2)+1e-8) + decay;
        double proposed = model->parameters[i] - lr*update;
        if (!isfinite(proposed) || fabs(proposed) > FLT_MAX || v > FLT_MAX) return 0;
        s->proposal[i] = (float)proposed;
    }
    /* Commit only after every proposed weight and moment has passed validation. */
    for (size_t i = 0; i < n; ++i) {
        double g = s->grad[i]*scale;
        s->m[i] = (float)(0.9*s->m[i]+0.1*g);
        s->v[i] = (float)(0.999*s->v[i]+0.001*g*g);
    }
    memcpy(model->parameters, s->proposal, n*sizeof(float));
    ++model->steps;
    s->length = 0;
    gpt_zero_grad(model);
    if (grad_norm) *grad_norm = norm;
    return 1;
}

static int test_causality(Gpt *model)
{
    const int input[5] = {0,1,2,3,4}, changed[5] = {0,1,2,8,9};
    float prefix[3][11], row[5];
    if (!gpt_forward(model, input, 5)) return 0;
    for (int t = 0; t < 3; ++t) memcpy(prefix[t], gpt_logits(model,t), sizeof(prefix[t]));
    for (int l = 0; l < model->config.layers; ++l)
        for (int h = 0; h < model->config.heads; ++h)
            for (int t = 0; t < 5; ++t) {
                if (!gpt_attention_row(model,l,h,t,row)) return 0;
                double sum = 0.0;
                for (int j = 0; j < 5; ++j) {
                    if (!isfinite(row[j]) || row[j] < 0.0f || (j > t && row[j] != 0.0f))
                        return 0;
                    sum += row[j];
                }
                if (fabs(sum-1.0) > 1e-6) return 0;
            }
    if (!gpt_forward(model, changed, 5)) return 0;
    for (int t = 0; t < 3; ++t)
        if (memcmp(prefix[t], gpt_logits(model,t), sizeof(prefix[t])) != 0) return 0;
    if (!gpt_forward(model, input, 3)) return 0;
    for (int t = 0; t < 3; ++t)
        if (memcmp(prefix[t], gpt_logits(model,t), sizeof(prefix[t])) != 0) return 0;
    return 1;
}

static int test_gradients(Gpt *model)
{
    const int input[5] = {2,7,1,2,4}, target[5] = {7,1,2,4,9};
    State *s = model->state;
    if (!gpt_forward(model, input, 5) || !isfinite(gpt_backward(model,target))) return 0;
    double max_error = 0.0;
    /* Central differences cover every scalar, including repeated input
       embeddings, all attention heads and the tied output contribution. */
    for (size_t i = 0; i < model->parameter_count; ++i) {
        float original = model->parameters[i];
        float hi = original + 0.001f, lo = original - 0.001f;
        model->parameters[i] = hi;
        if (!gpt_forward(model,input,5)) return 0;
        double plus = gpt_loss(model,target);
        model->parameters[i] = lo;
        if (!gpt_forward(model,input,5)) return 0;
        double minus = gpt_loss(model,target);
        model->parameters[i] = original;
        double numerical = (plus-minus)/((double)hi-lo);
        double analytical = s->grad[i]/5.0;
        double err = fabs(numerical-analytical);
        if (!isfinite(numerical) || !isfinite(analytical) ||
            err > 2e-4 + 0.004*fmax(fabs(numerical), fabs(analytical))) {
            fprintf(stderr,"Gradient %zu: analytisch %.8g, numerisch %.8g\n",i,analytical,numerical);
            return 0;
        }
        if (err > max_error) max_error = err;
    }
    printf("Gradiententest: %zu Parameter, maximale absolute Abweichung %.3g.\n",
           model->parameter_count, max_error);
    return 1;
}

static int test_optimizer(Gpt *model)
{
    /* Gradients from length 5 are still present; add a second sequence of
       different length and check token weighting, clipping and matrix decay. */
    const int input[2] = {3,2}, target[2] = {2,5};
    State *s = model->state;
    if (!gpt_forward(model,input,2) || !isfinite(gpt_backward(model,target)) ||
        s->accumulated != 7) return 0;
    size_t n = model->parameter_count;
    float *expected = malloc(2*n*sizeof(float));
    if (!expected) return 0;
    double norm2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double g = s->grad[i]/7.0;
        norm2 += g*g;
    }
    double norm = sqrt(norm2), reported = 0.0, lr = 0.002, wd = 0.125;
    for (size_t i = 0; i < n; ++i) {
        double clipped = s->grad[i]/21.0;
        double decay = i < s->matrices ? wd * model->parameters[i] : 0.0;
        expected[i] = (float)(model->parameters[i] - lr*(clipped/(fabs(clipped)+1e-8)+decay));
        expected[n+i] = (float)(0.1*clipped);
    }
    int ok = gpt_step(model, lr, wd, norm/3.0, &reported) && model->steps == 1 &&
             s->accumulated == 0 && isfinite(reported) && fabs(norm-reported) < 1e-10;
    for (size_t i = 0; ok && i < n; ++i)
        if (fabs(model->parameters[i]-expected[i]) > 1e-7 ||
            fabs(s->m[i]-expected[n+i]) > 1e-7 || s->grad[i] != 0.0f) ok = 0;
    free(expected);
    return ok;
}

static int test_stable_loss(Gpt *model)
{
    int token = 0, target = 1;
    if (!gpt_forward(model, &token, 1)) return 0;
    State *s = model->state;
    for (int j = 0; j < model->config.vocab_size; ++j) s->logits[j] = -10000.0f;
    s->logits[0] = s->logits[2] = 10000.0f;
    double loss = gpt_loss(model, &target);
    if (!isfinite(loss) || fabs(loss - (20000.0 + log(2.0))) > 1e-10) return 0;
    for (int j = 0; j < model->config.vocab_size; ++j) s->logits[j] = 1e30f;
    loss = gpt_loss(model, &target);
    return isfinite(loss) && fabs(loss-log((double)model->config.vocab_size)) < 1e-10;
}

int gpt_selftest(void)
{
    Gpt model = {0};
    GptConfig tiny = {11,5,6,2,2}, full = {50257,1024,768,12,12};
    if (gpt_parameter_count(full) != 124439808 || !gpt_init(&model,tiny,17)) {
        fprintf(stderr,"Modellform oder Initialisierung fehlgeschlagen.\n");
        return 0;
    }
    /* Less symmetric logits make query/key derivatives large enough to test
       with finite differences at float32 precision. */
    for (size_t i = 0; i < model.state->matrices; ++i) model.parameters[i] *= 12.0f;
    int ok = 0;
    if (!test_causality(&model)) fprintf(stderr,"Kausalitaetstest fehlgeschlagen.\n");
    else if (!test_gradients(&model)) fprintf(stderr,"Gradiententest fehlgeschlagen.\n");
    else if (!test_optimizer(&model)) fprintf(stderr,"AdamW-/Clipping-Test fehlgeschlagen.\n");
    else if (!test_stable_loss(&model)) fprintf(stderr,"Softmax-Test fehlgeschlagen.\n");
    else {
        puts("Kausale Maske, Prefix-Logits, Softmax, Token-Mittelung, Clipping und AdamW geprueft.");
        ok = 1;
    }
    gpt_free(&model);
    return ok;
}
