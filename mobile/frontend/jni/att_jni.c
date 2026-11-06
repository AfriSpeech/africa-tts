/* JNI binding of africa_tts_frontend for Android (class org.afrispeech.africatts.AfricaTtsFrontend).
 * Strings cross the boundary as UTF-8 byte arrays (JNI's NewStringUTF uses modified UTF-8). */
#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "africa_tts_frontend.h"

static char *dup_bytes(JNIEnv *env, jbyteArray a) {
  if (!a) return NULL;
  jsize n = (*env)->GetArrayLength(env, a);
  char *s = malloc((size_t)n + 1);
  if (!s) return NULL;
  (*env)->GetByteArrayRegion(env, a, 0, n, (jbyte *)s);
  s[n] = 0;
  return s;
}

static jbyteArray to_bytes(JNIEnv *env, const char *s) {
  jsize n = (jsize)strlen(s);
  jbyteArray a = (*env)->NewByteArray(env, n);
  if (a) (*env)->SetByteArrayRegion(env, a, 0, n, (const jbyte *)s);
  return a;
}

static void throw_ex(JNIEnv *env, const char *msg) {
  jclass c = (*env)->FindClass(env, "java/lang/IllegalStateException");
  if (c) (*env)->ThrowNew(env, c, msg ? msg : "africa_tts_frontend error");
}

JNIEXPORT jlong JNICALL Java_org_afrispeech_africatts_AfricaTtsFrontend_nativeCreate(
    JNIEnv *env, jclass cls, jbyteArray g2p_dir, jbyteArray espeak_dir) {
  (void)cls;
  char *g = dup_bytes(env, g2p_dir), *e = dup_bytes(env, espeak_dir), *err = NULL;
  att_frontend *f = att_create(g, e, &err);
  free(g);
  free(e);
  if (!f) {
    throw_ex(env, err);
    free(err);
    return 0;
  }
  return (jlong)(intptr_t)f;
}

JNIEXPORT jbyteArray JNICALL Java_org_afrispeech_africatts_AfricaTtsFrontend_nativeToIpa(
    JNIEnv *env, jclass cls, jlong ptr, jbyteArray lang, jbyteArray text) {
  (void)cls;
  att_frontend *f = (att_frontend *)(intptr_t)ptr;
  char *l = dup_bytes(env, lang), *t = dup_bytes(env, text);
  char *ipa = att_to_ipa(f, l, t);
  free(l);
  free(t);
  if (!ipa) {
    throw_ex(env, att_last_error(f));
    return NULL;
  }
  jbyteArray out = to_bytes(env, ipa);
  att_free(ipa);
  return out;
}

JNIEXPORT void JNICALL Java_org_afrispeech_africatts_AfricaTtsFrontend_nativeDestroy(
    JNIEnv *env, jclass cls, jlong ptr) {
  (void)env;
  (void)cls;
  att_destroy((att_frontend *)(intptr_t)ptr);
}
