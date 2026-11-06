/*
 * africa_tts_frontend: text -> the IPA that Africa TTS was trained on, in C.
 *
 * For on-device use next to sherpa-onnx (Android, iOS, desktop). The model reads IPA, so
 * every input goes through this front end before it reaches sherpa-onnx:
 *   - African languages: africa-g2p's C library (afrig2p), one <lang>.afg2p table each,
 *     converted word by word exactly as the training data was.
 *   - English ("eng"): espeak-ng (en-us voice) through its C API, post-processed exactly like
 *     training did (phonemizer: no stress, language-switch flags removed; africa-g2p's
 *     normalisation: ASCII g -> ɡ, affricate tie bars), word by word with leading/trailing
 *     punctuation kept.
 *
 * English needs espeak-ng (GPL-3.0-or-later): an app that links it must comply with the GPL.
 * Pass espeak_data_dir = NULL to build without English.
 *
 * Not thread-safe: use one handle per thread (espeak-ng itself is global state).
 */
#ifndef AFRICA_TTS_FRONTEND_H
#define AFRICA_TTS_FRONTEND_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct att_frontend att_frontend;

/* g2p_dir: folder holding <lang>.afg2p tables (loaded lazily, on first use of a language).
 * espeak_data_dir: folder that CONTAINS "espeak-ng-data" (espeak-ng's convention), or NULL
 * to disable English. On failure returns NULL and stores a malloc'd message in *error. */
att_frontend *att_create(const char *g2p_dir, const char *espeak_data_dir, char **error);
void att_destroy(att_frontend *f);

/* Convert UTF-8 text in language `lang` (ISO 639-3 code as used by the model, e.g. "twi",
 * "eng") to IPA. Returns a malloc'd UTF-8 string (release with att_free), or NULL on error
 * (unknown language, missing table, invalid UTF-8); see att_last_error. */
char *att_to_ipa(att_frontend *f, const char *lang, const char *text);

const char *att_last_error(const att_frontend *f);
void att_free(void *p);

#ifdef __cplusplus
}
#endif

#endif /* AFRICA_TTS_FRONTEND_H */
