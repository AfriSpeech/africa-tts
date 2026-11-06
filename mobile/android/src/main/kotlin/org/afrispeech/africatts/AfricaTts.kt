package org.afrispeech.africatts

import com.k2fsa.sherpa.onnx.GeneratedAudio
import com.k2fsa.sherpa.onnx.GenerationConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsPocketModelConfig
import com.k2fsa.sherpa.onnx.WaveReader
import java.io.File

/**
 * Africa TTS on device: africa-g2p / espeak-ng front end + sherpa-onnx Pocket TTS.
 *
 * [modelDir] is an unpacked model bundle on the file system (see mobile/README.md):
 *   lm_main.int8.onnx lm_flow.int8.onnx encoder.onnx decoder.int8.onnx text_conditioner.onnx
 *   vocab.json token_scores.json  g2p/<lang>.afg2p  voices/<lang>.wav  [espeak-ng-data/]
 * Runs either bundle: Africa TTS (24 layers, unguided) or Africa TTS Turbo (6 layers, guidance
 * distilled in). The guided 24-layer mode needs a second pass per frame and is Python-only.
 */
class AfricaTts(private val modelDir: String, numThreads: Int = 2) : AutoCloseable {
    private val frontend = AfricaTtsFrontend(
        g2pDir = "$modelDir/g2p",
        espeakDir = modelDir.takeIf { File("$it/espeak-ng-data").isDirectory },
    )
    private val tts = OfflineTts(
        config = OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                pocket = OfflineTtsPocketModelConfig(
                    lmMain = "$modelDir/lm_main.int8.onnx",
                    lmFlow = "$modelDir/lm_flow.int8.onnx",
                    encoder = "$modelDir/encoder.onnx",
                    decoder = "$modelDir/decoder.int8.onnx",
                    textConditioner = "$modelDir/text_conditioner.onnx",
                    vocabJson = "$modelDir/vocab.json",
                    tokenScoresJson = "$modelDir/token_scores.json",
                ),
                numThreads = numThreads,
            ),
        ),
    )
    private val voices = HashMap<String, Pair<FloatArray, Int>>()

    val sampleRate: Int get() = tts.sampleRate()

    /** The bundled default voice for [lang] (3.5-6 s of speech in that language). */
    fun defaultVoice(lang: String): Pair<FloatArray, Int> = voices.getOrPut(lang) {
        val wave = WaveReader.readWave(filename = "$modelDir/voices/$lang.wav")
        wave.samples to wave.sampleRate
    }

    /**
     * Speak [text] (normal spelling) in [lang] (ISO 639-3, e.g. "twi", "eng"), in the language's
     * default voice or in [voice] = (samples, sampleRate) of 3-10 s of clean speech to clone.
     * Settings match pocket-tts: temperature 0.3, one flow step, no pause compression.
     */
    fun speak(text: String, lang: String, voice: Pair<FloatArray, Int>? = null): GeneratedAudio {
        val (samples, sr) = voice ?: defaultVoice(lang)
        val config = GenerationConfig(
            referenceAudio = samples,
            referenceSampleRate = sr,
            numSteps = 1,
            silenceScale = 1.0f,
            extra = mapOf("temperature" to "0.3"),
        )
        return tts.generateWithConfig(frontend.toIpa(lang, text), config)
    }

    override fun close() {
        tts.release()
        frontend.close()
    }
}
