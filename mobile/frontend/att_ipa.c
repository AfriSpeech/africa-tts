/* att_ipa: read UTF-8 lines on stdin, write the model's IPA on stdout.
 *   att_ipa -g G2P_DIR [-e ESPEAK_DIR] -l LANG
 * ESPEAK_DIR contains espeak-ng-data (omit to use espeak-ng's default location).
 * With -t, each input line is "LANG<TAB>TEXT" instead and -l is not needed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "africa_tts_frontend.h"

int main(int argc, char **argv) {
  const char *g2p = ".", *espeak = NULL, *lang = NULL;
  int tabbed = 0, english = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-g") && i + 1 < argc) g2p = argv[++i];
    else if (!strcmp(argv[i], "-e") && i + 1 < argc) espeak = argv[++i];
    else if (!strcmp(argv[i], "-l") && i + 1 < argc) lang = argv[++i];
    else if (!strcmp(argv[i], "-t")) tabbed = 1;
    else if (!strcmp(argv[i], "--no-english")) english = 0;
    else {
      fprintf(stderr, "usage: %s -g G2P_DIR [-e ESPEAK_DIR] (-l LANG | -t) [--no-english]\n", argv[0]);
      return 2;
    }
  }
  if (!lang && !tabbed) { fprintf(stderr, "need -l LANG or -t\n"); return 2; }
  char *err = NULL;
  /* espeak-ng treats a NULL path as "default location"; "" keeps English on with the default */
  att_frontend *f = att_create(g2p, english ? (espeak ? espeak : "") : NULL, &err);
  if (!f && english && !espeak) f = att_create(g2p, NULL, &err);
  if (!f) { fprintf(stderr, "error: %s\n", err ? err : "?"); return 1; }
  char *line = NULL;
  size_t cap = 0;
  ssize_t n;
  int rc = 0;
  while ((n = getline(&line, &cap, stdin)) >= 0) {
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
    const char *l = lang, *text = line;
    if (tabbed) {
      char *tab = strchr(line, '\t');
      if (!tab) { puts(""); continue; }
      *tab = 0;
      l = line;
      text = tab + 1;
    }
    char *ipa = att_to_ipa(f, l, text);
    if (!ipa) { fprintf(stderr, "error: %s\n", att_last_error(f)); puts(""); rc = 1; continue; }
    puts(ipa);
    att_free(ipa);
  }
  free(line);
  att_destroy(f);
  return rc;
}
