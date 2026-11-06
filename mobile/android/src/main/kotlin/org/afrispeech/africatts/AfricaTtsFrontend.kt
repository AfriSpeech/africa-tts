package org.afrispeech.africatts

/**
 * Text -> the IPA Africa TTS was trained on (africa-g2p for African languages, espeak-ng 1.51
 * for English). Wraps libafrica_tts_frontend_jni.so (mobile/frontend).
 *
 * @param g2pDir folder with the <lang>.afg2p tables (hf://AfriSpeech/africa-tts/g2p/)
 * @param espeakDir folder that CONTAINS espeak-ng-data, or null to disable English
 */
class AfricaTtsFrontend(g2pDir: String, espeakDir: String?) : AutoCloseable {
    private var ptr: Long = nativeCreate(g2pDir.toByteArray(), espeakDir?.toByteArray())

    /** @param lang ISO 639-3 code as used by the model, e.g. "twi", "hau", "eng". */
    fun toIpa(lang: String, text: String): String {
        check(ptr != 0L) { "AfricaTtsFrontend is closed" }
        return String(nativeToIpa(ptr, lang.toByteArray(), text.toByteArray()), Charsets.UTF_8)
    }

    override fun close() {
        if (ptr != 0L) {
            nativeDestroy(ptr)
            ptr = 0L
        }
    }

    companion object {
        init {
            System.loadLibrary("africa_tts_frontend_jni")
        }

        @JvmStatic private external fun nativeCreate(g2pDir: ByteArray, espeakDir: ByteArray?): Long
        @JvmStatic private external fun nativeToIpa(ptr: Long, lang: ByteArray, text: ByteArray): ByteArray
        @JvmStatic private external fun nativeDestroy(ptr: Long)
    }
}
