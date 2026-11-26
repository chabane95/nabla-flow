/* Prepare contiguous Tiny Shakespeare splits for the byte-model CLI.
   Usage: shakespeare input.txt new-output-directory */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#define new_directory(p) _mkdir(p)
#else
#include <sys/stat.h>
#define new_directory(p) mkdir(p, 0777)
#endif

enum { TRAIN_END = 800, VAL_END = 864, TEST_END = 928 };

static int subset(int paragraph)
{
    return paragraph < TRAIN_END ? 0 : paragraph < VAL_END ? 1 : 2;
}

static FILE *output_file(const char *directory, const char *name)
{
    size_t n = strlen(directory) + strlen(name) + 2;
    char *path = malloc(n);
    if (!path) return NULL;
    snprintf(path,n,"%s/%s",directory,name);
    FILE *fp = fopen(path,"wb");
    free(path);
    return fp;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fputs("Usage: shakespeare input.txt NEW-DIRECTORY\n",stderr);
        return EXIT_FAILURE;
    }
    FILE *in = fopen(argv[1],"rb");
    if (!in) { perror(argv[1]); return EXIT_FAILURE; }
    long size = -1;
    if (fseek(in,0,SEEK_END) == 0) size = ftell(in);
    unsigned char *data = size > 0 ? malloc((size_t)size) : NULL;
    int ok = data && fseek(in,0,SEEK_SET) == 0 && fread(data,1,(size_t)size,in) == (size_t)size;
    if (fclose(in)) ok = 0;
    if (!ok || new_directory(argv[2]) != 0) {
        fputs("Readable source and a new output directory required.\n",stderr);
        free(data);
        return EXIT_FAILURE;
    }
    FILE *out[3] = {output_file(argv[2],"train.txt"), output_file(argv[2],"val.txt"),
                    output_file(argv[2],"test.txt")};
    int paragraph = 0, in_paragraph = 0;
    ok = out[0] && out[1] && out[2];
    for (size_t pos = 0; ok && pos < (size_t)size && paragraph < TEST_END;) {
        size_t start = pos;
        while (pos < (size_t)size && data[pos] != '\n' && data[pos] != '\r') ++pos;
        size_t n = pos - start;
        if (pos < (size_t)size && data[pos] == '\r') ++pos;
        if (pos < (size_t)size && data[pos] == '\n') ++pos;
        FILE *f = out[subset(paragraph)];
        if (n) {
            if (in_paragraph) ok = fputc(' ',f) != EOF;
            ok = ok && fwrite(data+start,1,n,f) == n;
            in_paragraph = 1;
        } else if (in_paragraph) {
            ok = fputc('\n',f) != EOF;
            ++paragraph;
            in_paragraph = 0;
        }
    }
    if (ok && in_paragraph) {
        ok = fputc('\n',out[subset(paragraph)]) != EOF;
        ++paragraph;
    }
    for (int k = 0; k < 3; ++k)
        if (out[k] && fclose(out[k])) ok = 0;
    free(data);
    if (!ok || paragraph != TEST_END) {
        fputs("Split failed: check source length and file permissions.\n",stderr);
        return EXIT_FAILURE;
    }
    printf("Contiguous paragraphs: train 1-%d, validation %d-%d, test %d-%d.\n",
           TRAIN_END,TRAIN_END+1,VAL_END,VAL_END+1,TEST_END);
    return EXIT_SUCCESS;
}
