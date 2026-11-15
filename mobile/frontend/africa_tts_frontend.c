/*
 * africa_tts_frontend.c - see africa_tts_frontend.h.
 *
 * The English path reproduces, step by step, what training ran
 * (recipes/africa/data/to_ipa.py -> africa_g2p.english -> phonemizer 3.4 EspeakBackend("en-us",
 * with_stress=False, language_switch="remove-flags"), one word at a time):
 *   1. split leading/trailing punctuation off the word (Python r"^(\W*)(.*?)(\W*)$")
 *   2. phonemizer preprocessing: punctuation marks ;:,.!?¡¿—…"«»“”(){}[] -> space (except
 *      '.' and ',' between two digits), strip
 *   3. espeak_TextToPhonemes(UTF-8, IPA, phoneme separator '_') until the text is consumed,
 *      clause outputs joined with ' '
 *   4. phonemizer postprocessing: strip, '\n' -> ' ', "  " -> " ", '_'+ -> '_', "_ " -> " ",
 *      remove (xx) language flags, per word remove [ˈˌ'-]+ and '_'
 *   5. africa-g2p normalisation: g -> ɡ, drop ˈ ˌ |, tʃ/ʧ -> t͡ʃ, dʒ/ʤ -> d͡ʒ
 *   6. remove spaces (one word in, one IPA word out), re-attach the punctuation
 */
#include "africa_tts_frontend.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "afrig2p.h"

#ifndef ATT_NO_ESPEAK
#include <espeak-ng/speak_lib.h>
#endif

#define ATT_MAX_LANGS 256

typedef struct {
  char code[16];
  afrig2p *g;
} lang_slot;

struct att_frontend {
  char *g2p_dir;
  int english;
  char *codeswitch_data;
  size_t codeswitch_len;
  lang_slot langs[ATT_MAX_LANGS];
  int n_langs;
  char error[512];
};

/* ---------------------------------------------------------------- string buffer */

typedef struct {
  char *s;
  size_t n, cap;
} buf;

static int buf_reserve(buf *b, size_t extra) {
  if (b->n + extra + 1 <= b->cap) return 0;
  size_t cap = b->cap ? b->cap : 64;
  while (cap < b->n + extra + 1) cap *= 2;
  char *s = realloc(b->s, cap);
  if (!s) return -1;
  b->s = s;
  b->cap = cap;
  return 0;
}

static int buf_add(buf *b, const char *s, size_t len) {
  if (buf_reserve(b, len)) return -1;
  memcpy(b->s + b->n, s, len);
  b->n += len;
  b->s[b->n] = 0;
  return 0;
}

static int buf_adds(buf *b, const char *s) { return buf_add(b, s, strlen(s)); }

static char *buf_take(buf *b) {
  if (!b->s && buf_reserve(b, 0)) return NULL;
  b->s[b->n] = 0;
  return b->s;
}

/* Replace every occurrence of `from` by `to` (to may be ""). */
static char *replace_all(char *s, const char *from, const char *to) {
  size_t lf = strlen(from);
  if (!strstr(s, from)) return s;
  buf out = {0};
  for (char *p = s; *p;) {
    if (strncmp(p, from, lf) == 0) {
      buf_adds(&out, to);
      p += lf;
    } else {
      buf_add(&out, p++, 1);
    }
  }
  free(s);
  return buf_take(&out);
}

static void set_error(att_frontend *f, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(f->error, sizeof f->error, fmt, ap);
  va_end(ap);
}

/* ---------------------------------------------------------------- UTF-8 helpers */

/* Decode one code point; returns its byte length (>= 1), or 0 on invalid UTF-8. */
static int utf8_decode(const unsigned char *s, unsigned *cp) {
  if (s[0] < 0x80) { *cp = s[0]; return 1; }
  if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) {
    *cp = ((s[0] & 0x1F) << 6) | (s[1] & 0x3F);
    return 2;
  }
  if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
    *cp = ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    return 3;
  }
  if ((s[0] & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 &&
      (s[3] & 0xC0) == 0x80) {
    *cp = ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    return 4;
  }
  return 0;
}

/* Python's \w (str.isalnum() or '_') for the characters English text actually contains:
 * ASCII exactly; non-ASCII letters/digits are word characters, while the punctuation and
 * symbol blocks below are not. */
static int is_word_char(unsigned cp) {
  if (cp < 0x80) return isalnum((int)cp) || cp == '_';
  if (cp >= 0x00A0 && cp <= 0x00BF) return cp == 0xAA || cp == 0xB2 || cp == 0xB3 ||
                                           cp == 0xB5 || cp == 0xB9 || cp == 0xBA ||
                                           (cp >= 0xBC && cp <= 0xBE);
  if (cp == 0x00D7 || cp == 0x00F7) return 0;
  if (cp >= 0x0300 && cp <= 0x036F) return 0;  /* combining marks: not alnum in Python */
  if (cp >= 0x2000 && cp <= 0x206F) return 0;  /* general punctuation: ‘ ’ “ ” — … etc. */
  if (cp >= 0x20A0 && cp <= 0x20CF) return 0;  /* currency */
  if (cp >= 0x2100 && cp <= 0x2BFF) return 0;  /* letterlike..misc symbols, arrows, math */
  if (cp >= 0x3000 && cp <= 0x303F) return 0;  /* CJK punctuation */
  if (cp >= 0xFE30 && cp <= 0xFE6F) return 0;
  if (cp >= 0xFF00 && cp <= 0xFF0F) return 0;
  if (cp >= 0x1F000) return 0;                 /* emoji and pictographs */
  return 1;
}

/* ---------------------------------------------------------------- African languages */

static afrig2p *get_table(att_frontend *f, const char *lang) {
  for (int i = 0; i < f->n_langs; i++)
    if (strcmp(f->langs[i].code, lang) == 0) return f->langs[i].g;
  if (strlen(lang) >= sizeof f->langs[0].code || strchr(lang, '/') || strchr(lang, '.')) {
    set_error(f, "invalid language code '%s'", lang);
    return NULL;
  }
  if (f->n_langs >= ATT_MAX_LANGS) {
    set_error(f, "too many languages loaded");
    return NULL;
  }
  buf path = {0};
  buf_adds(&path, f->g2p_dir);
  buf_adds(&path, "/");
  buf_adds(&path, lang);
  buf_adds(&path, ".afg2p");
  char *err = NULL;
  afrig2p *g = afrig2p_create_from_file(path.s, &err);
  if (!g) {
    set_error(f, "cannot load %s: %s", path.s, err ? err : "unknown error");
    afrig2p_free(err);
    free(path.s);
    return NULL;
  }
  free(path.s);
  if (!afrig2p_has_ipa(g)) {
    set_error(f, "language '%s' has no IPA table", lang);
    afrig2p_destroy(g);
    return NULL;
  }
  snprintf(f->langs[f->n_langs].code, sizeof f->langs[0].code, "%s", lang);
  if (f->codeswitch_data) {
    afrig2p_set_english_words(g, f->codeswitch_data, f->codeswitch_len);
  }
  f->langs[f->n_langs++].g = g;
  return g;
}

/* African language conversion via afrig2p, with code-switching lexicon applied if present. */
static char *african_to_ipa(att_frontend *f, const char *lang, const char *text) {
  afrig2p *g = get_table(f, lang);
  if (!g) return NULL;
  char *ipa = afrig2p_to_ipa(g, text, "");
  if (!ipa) {
    set_error(f, "invalid UTF-8 input");
    return NULL;
  }
  return ipa;
}

/* ---------------------------------------------------------------- English */

#ifndef ATT_NO_ESPEAK

static const char *const MARKS[] = {";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"",
                                    "«", "»", "“", "”", "(", ")", "{", "}", "[", "]"};

/* phonemizer Punctuation.remove: marks -> ' ' (not '.'/',' between two digits), strip. */
static char *remove_marks(const char *s) {
  buf out = {0};
  size_t n = strlen(s);
  for (size_t i = 0; i < n;) {
    int hit = 0;
    for (size_t m = 0; m < sizeof MARKS / sizeof *MARKS; m++) {
      size_t len = strlen(MARKS[m]);
      if (strncmp(s + i, MARKS[m], len) == 0) {
        int decimal = (s[i] == '.' || s[i] == ',') && i > 0 && isdigit((unsigned char)s[i - 1]) &&
                      isdigit((unsigned char)s[i + 1]);
        if (!decimal) {
          buf_adds(&out, " ");
          i += len;
          hit = 1;
        }
        break;
      }
    }
    if (!hit) buf_add(&out, s + i++, 1);
  }
  char *r = buf_take(&out);
  /* strip */
  char *a = r;
  while (*a && isspace((unsigned char)*a)) a++;
  char *z = a + strlen(a);
  while (z > a && isspace((unsigned char)z[-1])) z--;
  *z = 0;
  memmove(r, a, (size_t)(z - a) + 1);
  return r;
}

/* Steps 2-5 for one punctuation-free word core. */
static char *espeak_word(const char *core) {
  char *text = remove_marks(core);
  buf raw = {0};
  if (*text) {
    const void *ptr = text;
    const int mode = ('_' << 8) | espeakPHONEMES_IPA;
    while (ptr) {
      const char *ph = espeak_TextToPhonemes(&ptr, espeakCHARS_UTF8, mode);
      if (ph && *ph) {
        if (raw.n) buf_adds(&raw, " ");
        buf_adds(&raw, ph);
      }
    }
  }
  free(text);
  char *s = buf_take(&raw);
  /* strip, '\n' -> ' ', "  " -> " " (one pass, as Python str.replace) */
  s = replace_all(s, "\n", " ");
  s = replace_all(s, "  ", " ");
  /* '_'+ -> '_', then "_ " -> " " */
  while (strstr(s, "__")) s = replace_all(s, "__", "_");
  s = replace_all(s, "_ ", " ");
  /* remove-flags: \(.+?\) */
  buf noflags = {0};
  for (char *p = s; *p;) {
    /* '(' + at least one character (which may be ')') + the next ')' */
    char *close = (*p == '(' && p[1]) ? strchr(p + 2, ')') : NULL;
    if (close) {
      p = close + 1;
    } else {
      buf_add(&noflags, p++, 1);
    }
  }
  free(s);
  s = buf_take(&noflags);
  /* stress [ˈˌ'-]+ and the '_' phone separator (separator.phone = ""), then africa-g2p */
  static const char *const DROP[] = {"ˈ", "ˌ", "'", "-", "_", "|"};
  for (size_t i = 0; i < sizeof DROP / sizeof *DROP; i++) s = replace_all(s, DROP[i], "");
  s = replace_all(s, "g", "ɡ");
  s = replace_all(s, "tʃ", "t͡ʃ");
  s = replace_all(s, "dʒ", "d͡ʒ");
  s = replace_all(s, "ʧ", "t͡ʃ");
  s = replace_all(s, "ʤ", "d͡ʒ");
  /* one word in, one IPA word out */
  s = replace_all(s, " ", "");
  s = replace_all(s, "\t", "");
  return s;
}

static char *english_to_ipa(att_frontend *f, const char *text) {
  if (!f->english) {
    set_error(f, "English is disabled (no espeak-ng data directory given)");
    return NULL;
  }
  buf out = {0};
  const char *p = text;
  while (*p) {
    while (*p && isspace((unsigned char)*p)) p++;
    if (!*p) break;
    const char *ws = p;
    while (*p && !isspace((unsigned char)*p)) p++;
    const char *we = p;
    /* step 1: leading and trailing non-word characters */
    const char *cs = ws;
    while (cs < we) {
      unsigned cp;
      int len = utf8_decode((const unsigned char *)cs, &cp);
      if (!len) { set_error(f, "invalid UTF-8 input"); free(out.s); return NULL; }
      if (is_word_char(cp)) break;
      cs += len;
    }
    const char *ce = cs;
    const char *last_word_end = cs;
    while (ce < we) {
      unsigned cp;
      int len = utf8_decode((const unsigned char *)ce, &cp);
      if (!len) { set_error(f, "invalid UTF-8 input"); free(out.s); return NULL; }
      ce += len;
      if (is_word_char(cp)) last_word_end = ce;
    }
    ce = last_word_end;
    if (out.n) buf_adds(&out, " ");
    buf_add(&out, ws, (size_t)(cs - ws));
    if (ce > cs) {
      char *core = strndup(cs, (size_t)(ce - cs));
      char *ipa = espeak_word(core);
      free(core);
      buf_adds(&out, ipa);
      free(ipa);
    }
    buf_add(&out, ce, (size_t)(we - ce));
  }
  return buf_take(&out);
}

#endif /* ATT_NO_ESPEAK */

/* ---------------------------------------------------------------- public API */

int att_set_codeswitch(att_frontend *f, const char *tsv_data, size_t len) {
  if (!f) return -1;
  free(f->codeswitch_data);
  f->codeswitch_data = NULL;
  f->codeswitch_len = 0;
  if (tsv_data && len > 0) {
    f->codeswitch_data = malloc(len + 1);
    if (!f->codeswitch_data) return -1;
    memcpy(f->codeswitch_data, tsv_data, len);
    f->codeswitch_data[len] = 0;
    f->codeswitch_len = len;
  }
  for (int i = 0; i < f->n_langs; i++) {
    afrig2p_set_english_words(f->langs[i].g, f->codeswitch_data, f->codeswitch_len);
  }
  return 0;
}

att_frontend *att_create(const char *g2p_dir, const char *espeak_data_dir, char **error) {
  att_frontend *f = calloc(1, sizeof *f);
  if (!f) return NULL;
  f->g2p_dir = strdup(g2p_dir ? g2p_dir : ".");

  /* Try to load default codeswitch.tsv from g2p_dir if present */
  buf cs_path = {0};
  buf_adds(&cs_path, f->g2p_dir);
  buf_adds(&cs_path, "/codeswitch.tsv");
  FILE *fp = fopen(cs_path.s, "rb");
  if (!fp) {
    cs_path.n = 0;
    buf_adds(&cs_path, f->g2p_dir);
    buf_adds(&cs_path, "/codeswitch_en.tsv");
    fp = fopen(cs_path.s, "rb");
  }
  if (fp) {
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz > 0) {
      char *data = malloc(sz + 1);
      if (data && fread(data, 1, sz, fp) == (size_t)sz) {
        data[sz] = 0;
        att_set_codeswitch(f, data, (size_t)sz);
      }
      free(data);
    }
    fclose(fp);
  }
  free(cs_path.s);

  if (espeak_data_dir) {
#ifdef ATT_NO_ESPEAK
    if (error) *error = strdup("built without espeak-ng (ATT_NO_ESPEAK)");
    att_destroy(f);
    return NULL;
#else
    if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, espeak_data_dir, 0) < 0 ||
        espeak_SetVoiceByName("en-us") != EE_OK) {
      if (error) *error = strdup("espeak-ng initialisation failed (check espeak_data_dir)");
      att_destroy(f);
      return NULL;
    }
    f->english = 1;
#endif
  }
  return f;
}

void att_destroy(att_frontend *f) {
  if (!f) return;
  for (int i = 0; i < f->n_langs; i++) afrig2p_destroy(f->langs[i].g);
#ifndef ATT_NO_ESPEAK
  if (f->english) espeak_Terminate();
#endif
  free(f->codeswitch_data);
  free(f->g2p_dir);
  free(f);
}

/* Greek/Cyrillic lookalikes of African Latin letters (Twi "ԑ", Ewe "ͻ", ...): the G2P tables only
 * know the Latin letters. Kept in sync with LOOKALIKES in africa_tts/frontend.py. */
static const char *const LOOKALIKES[][2] = {
    {"ͻ", "ɔ"}, {"Ͻ", "Ɔ"}, {"ԑ", "ɛ"}, {"Ԑ", "Ɛ"}, {"ε", "ɛ"},
    {"Ε", "Ɛ"}, {"Γ", "Ɣ"}, {"γ", "ɣ"}, {"ῖ", "ĩ"},
};

char *att_to_ipa(att_frontend *f, const char *lang, const char *text) {
  f->error[0] = 0;
  if (!lang || !text) {
    set_error(f, "null argument");
    return NULL;
  }
  if (strcmp(lang, "eng") == 0) {
#ifdef ATT_NO_ESPEAK
    set_error(f, "built without espeak-ng (ATT_NO_ESPEAK)");
    return NULL;
#else
    return english_to_ipa(f, text);
#endif
  }
  size_t n = strlen(text);
  char *clean = malloc(n + 1);
  if (!clean) {
    set_error(f, "out of memory");
    return NULL;
  }
  memcpy(clean, text, n + 1);
  for (size_t i = 0; i < sizeof LOOKALIKES / sizeof LOOKALIKES[0]; i++)
    clean = replace_all(clean, LOOKALIKES[i][0], LOOKALIKES[i][1]);
  char *ipa = african_to_ipa(f, lang, clean);
  free(clean);
  return ipa;
}

const char *att_last_error(const att_frontend *f) { return f->error; }

void att_free(void *p) { free(p); }
