#include "checkpoint.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static const unsigned char magic[8] = {'N','F','G','P','T','0','1','\n'};
static const uint64_t offset_basis = UINT64_C(14695981039346656037);

static uint64_t hash_bytes(uint64_t hash, const unsigned char *p, size_t n)
{
    while (n--)
        hash = (hash ^ *p++) * UINT64_C(1099511628211);
    return hash;
}

static void pack(unsigned char *p, uint64_t n, unsigned width)
{
    for (unsigned i = 0; i < width; ++i) {
        p[i] = (unsigned char)n;
        n >>= 8;
    }
}

static uint64_t unpack(const unsigned char *p, unsigned width)
{
    uint64_t n = 0;
    while (width)
        n = (n << 8) | p[--width];
    return n;
}

static int ieee_supported(void)
{
    return CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 &&
           FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128;
}

static void pack_parameter(unsigned char *dest, const float *value)
{
    uint32_t bits;
    memcpy(&bits, value, sizeof(bits));
    pack(dest, bits, 4);
}

uint64_t checkpoint_parameter_hash(const Gpt *g)
{
    uint64_t hash = offset_basis;
    unsigned char bytes[4];
    for (size_t i = 0; i < g->parameter_count; ++i) {
        pack_parameter(bytes, g->parameters+i);
        hash = hash_bytes(hash, bytes, 4);
    }
    return hash;
}

/* Five u32 config fields, one u64 parameter count and 256 training-byte flags. */
enum { HEADER = 20 + 8 + 256, CHUNK = 8192 };

static int write_model(FILE *f, const Gpt *g, const unsigned char seen[256])
{
    const GptConfig c = g->config;
    const int values[5] = {c.vocab_size,c.context,c.width,c.heads,c.layers};
    unsigned char header[HEADER] = {0}, buf[4 * CHUNK];
    for (int i = 0; i < 5; ++i)
        pack(header + 4*i, (uint32_t)values[i], 4);
    pack(header + 20, g->parameter_count, 8);
    if (seen)
        memcpy(header + 28, seen, 256);
    uint64_t hash = hash_bytes(offset_basis, header, sizeof(header));
    if (fwrite(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fwrite(header, 1, sizeof(header), f) != sizeof(header))
        return 0;
    for (size_t pos = 0; pos < g->parameter_count;) {
        size_t n = g->parameter_count - pos;
        if (n > CHUNK) n = CHUNK;
        for (size_t k = 0; k < n; ++k) {
            if (!isfinite(g->parameters[pos+k]))
                return 0;
            pack_parameter(buf + 4*k, g->parameters+pos+k);
        }
        if (fwrite(buf, 4, n, f) != n)
            return 0;
        hash = hash_bytes(hash, buf, 4*n);
        pos += n;
    }
    pack(buf, hash, 8);
    return fwrite(buf, 1, 8, f) == 8;
}

static int read_model(FILE *f, Gpt *g, unsigned char seen[256])
{
    unsigned char header[HEADER], buf[4*CHUNK];
    if (fread(buf, 1, sizeof(magic), f) != sizeof(magic) || memcmp(buf, magic, sizeof(magic)) ||
        fread(header, 1, sizeof(header), f) != sizeof(header))
        return 0;
    int v[5];
    for (int i = 0; i < 5; ++i) {
        uint64_t a = unpack(header + 4*i, 4);
        if (a > INT_MAX)
            return 0;
        v[i] = (int)a;
    }
    uint64_t count = unpack(header + 20, 8);
    GptConfig config = {v[0],v[1],v[2],v[3],v[4]};
    uint64_t hash = hash_bytes(offset_basis, header, sizeof(header));
    long pos = ftell(f), end;
    if (!count || count != gpt_parameter_count(config) ||
        pos < 0 || fseek(f, 0, SEEK_END) || (end = ftell(f)) < pos + 8 ||
        (uint64_t)(end - pos - 8) / 4 != count || (end - pos - 8) % 4 ||
        fseek(f, pos, SEEK_SET) || !gpt_init(g, config, 0) || g->parameter_count != count)
        return 0;
    for (size_t p = 0; p < g->parameter_count;) {
        size_t n = g->parameter_count-p;
        if (n > CHUNK) n = CHUNK;
        if (fread(buf, 4, n, f) != n)
            return 0;
        hash = hash_bytes(hash, buf, 4*n);
        for (size_t k = 0; k < n; ++k) {
            uint32_t bits = (uint32_t)unpack(buf + k*4, 4);
            memcpy(&g->parameters[p+k], &bits, 4);
            if (!isfinite(g->parameters[p+k]))
                return 0;
        }
        p += n;
    }
    if (fread(buf, 1, 8, f) != 8 || unpack(buf, 8) != hash)
        return 0;
    if (seen) memcpy(seen, header + 28, 256);
    return 1;
}

int checkpoint_save(const char *path, const Gpt *g, const unsigned char seen[256])
{
    if (!ieee_supported() || !g || !g->parameters || !path || strlen(path) > SIZE_MAX - 16)
        return 0;
    size_t len = strlen(path), cap = len + 16;
    char *tmp = malloc(cap);
    if (!tmp) return 0;
    FILE *f = NULL;
    for (int i = 0; i < 32 && !f; ++i) {
        snprintf(tmp, cap, "%s.tmp.%d", path, i);
        f = fopen(tmp, "wbx");
    }
    int ok = 0;
    if (f) {
        ok = write_model(f, g, seen);
        if (fclose(f)) ok = 0;
        if (ok) {
#ifdef _WIN32
            ok = MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING) != 0;
#else
            ok = rename(tmp, path) == 0;
#endif
        }
        if (!ok) remove(tmp);
    }
    free(tmp);
    return ok;
}

int checkpoint_load(const char *path, Gpt *g, unsigned char seen[256])
{
    if (!ieee_supported() || !path || !g || g->parameters)
        return 0;
    Gpt candidate = {0};
    FILE *f = fopen(path, "rb");
    int ok = f && read_model(f, &candidate, seen);
    if (f && fclose(f)) ok = 0;
    if (ok)
        *g = candidate;
    else
        gpt_free(&candidate);
    return ok;
}

int checkpoint_selftest(void)
{
    Gpt a = {0}, b = {0};
    FILE *f = tmpfile();
    unsigned char alphabet[256] = {0}, loaded[256];
    alphabet[65] = 1;
    int ok = f && gpt_init(&a, (GptConfig){9,4,4,2,1}, 5) &&
             write_model(f,&a,alphabet) && !fseek(f,0,SEEK_SET) &&
             read_model(f,&b,loaded) && a.parameter_count == b.parameter_count &&
             !memcmp(a.parameters, b.parameters, a.parameter_count * sizeof(float)) &&
             !memcmp(alphabet,loaded,256);
    gpt_free(&b);
    if (ok) {
        ok = !fseek(f, -1, SEEK_END);
        int old = ok ? fgetc(f) : EOF;
        ok = old != EOF && !fseek(f, -1, SEEK_END) && fputc(old ^ 1,f) != EOF &&
             !fseek(f,0,SEEK_SET) && !read_model(f,&b,loaded);
    }
    if (f) fclose(f);
    gpt_free(&a);
    gpt_free(&b);
    printf("Checkpoint round-trip and damage detection: %s\n", ok ? "ok" : "FAILED");
    return ok;
}
