/* Rebuild the published UTF-8 corpus: corpus train|val|test|recall > file.
   The four arrangements of a pair always stay together in a split. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char *const words[2][10] = {
    {"rot", "blau", "grün", "gelb", "weiß", "rosa", "lila", "grau", "braun", "beige"},
    {"Glas", "Holz", "Sand", "Kork", "Gold", "Blei", "Erde", "Eis", "Harz", "Ton"}
};

int main(int argc, char **argv)
{
    if (argc != 2) return 1;
    const char *splits[] = {"train", "val", "test", "recall"};
    int mode = 0;
    while (mode < 4 && strcmp(argv[1], splits[mode])) ++mode;
    if (mode == 4) return 1;
    unsigned pair = 0;
    for (int a = 0; a < 10; ++a) for (int b = a + 1; b < 10; ++b, ++pair) {
        /* Permutation of 0..44. Exactly 31 / 7 / 7 unordered pairs. */
        unsigned slot = (17 * pair + 11) % 45;
        int split = slot < 31 ? 0 : slot < 38 ? 1 : 2;
        if (split != (mode == 3 ? 2 : mode)) continue;
        for (int family = 0; family < 2; ++family)
            for (int swap = 0; swap <= 1; ++swap)
                for (int ask = 0; ask <= 1; ++ask) {
                    const char *left = words[family][swap ? b : a];
                    const char *right = words[family][swap ? a : b];
                    if (family == 0)
                        printf("Links: %s. Rechts: %s. %s ist ", left, right, ask ? "Rechts" : "Links");
                    else
                        printf("Fach A: %s. Fach B: %s. In %s liegt ", left, right, ask ? "B" : "A");
                    if (mode == 3) putchar('\t');
                    printf("%s.\n", ask ? right : left);
                }
    }
    return ferror(stdout) ? 1 : 0;
}
