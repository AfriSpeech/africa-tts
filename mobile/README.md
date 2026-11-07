# Africa TTS on device (sherpa-onnx)

Africa TTS runs on phones and desktops through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)'s
Pocket TTS backend. The model reads **IPA**, so text goes through a small C front end first:

```
text ──► africa_tts_frontend ──► IPA ──► sherpa-onnx Pocket TTS ──► 24 kHz audio
          africa-g2p tables (92 African languages)        ▲
          espeak-ng 1.51 (English)                        └── voice prompt: voices/<lang>.wav (default) or your own
```

## Bundle

209 MB:

```bash
huggingface-cli download AfriSpeech/africa-tts --include "sherpa-onnx/*" --local-dir .
```

Contents: `lm_main.int8.onnx lm_flow.int8.onnx decoder.int8.onnx encoder.onnx text_conditioner.onnx
vocab.json token_scores.json g2p/ voices/ espeak-ng-data/`. int8 where it is safe; the encoder
stays fp32 because int8 distorts the voice conditioning.

## Generation settings (match pocket-tts)

| sherpa-onnx setting | value | why |
|---|---|---|
| `extra["temperature"]` | `"0.3"` | sherpa-onnx defaults to 0.7; the models were tuned for 0.3 |
| `num_steps` | `1` | pocket-tts default (Kotlin's `GenerationConfig` defaults to 5) |
| `silence_scale` | `1.0` | Kotlin's default 0.2 shortens pauses |
| reference audio | `voices/<lang>.wav` | 3.6-6 s, ends on a word boundary, like the training prompts |

## Front end (`frontend/`)

`africa_tts_frontend.{h,c}`: `att_create(g2p_dir, espeak_data_dir)`, `att_to_ipa(f, lang, text)`.
African languages use the [africa-g2p](https://github.com/AfriSpeech/africa-g2p) C library
(fetched by CMake at commit `76b53a8`, the one the models were trained with). English uses
espeak-ng's C API, post-processed step by step like training.

```bash
cmake -S mobile/frontend -B build/frontend && cmake --build build/frontend -j
echo "Akwaaba! Yɛfrɛ me Kofi." | build/frontend/att_ipa -g sherpa-onnx/g2p -e sherpa-onnx -l twi
python mobile/frontend/tests/check_parity.py build/frontend/att_ipa sherpa-onnx/g2p sherpa-onnx
```

**Parity with training** (tested on Linux): all 9,781 validation sentences of the 87 African
languages with validation data, and 146,578 of 146,580 distinct English training words (the two
misses are alignment artifacts), byte-identical to the IPA the model was trained on.
`tests/check_parity.py` re-checks any build against 5,442 reference rows: run it on every new
platform build.

**English needs espeak-ng 1.51.** espeak-ng 1.52 merged the vowels of "four" (`foːɹ`) and "for"
(`fɔːɹ`) in en-us, so about 2% of English words come out differently from training (the parity
check catches it). Build libespeak-ng from the [1.51 tag](https://github.com/espeak-ng/espeak-ng/tree/1.51)
(its `android/jni` has an NDK build) and use the bundle's `espeak-ng-data/` (English only,
820 KB). espeak-ng is GPL-3.0-or-later: an app that links it must comply with the GPL. To ship
without it, build with `-DATT_WITH_ESPEAK=OFF` and pass `espeak_data_dir = NULL`; the African
languages are unaffected.

## Android (`android/`)

- `AfricaTtsFrontend.kt`: Kotlin wrapper of the front end (JNI library
  `libafrica_tts_frontend_jni.so`, built by `frontend/CMakeLists.txt` when `ANDROID` is set).
- `AfricaTts.kt`: front end + sherpa-onnx `OfflineTts` with the settings above, default voices.

```kotlin
val tts = AfricaTts(modelDir = "${filesDir}/africa-tts")
val audio = tts.speak("Akwaaba! Yɛfrɛ me Kofi.", lang = "twi")    // default Twi voice
audio.save("${cacheDir}/out.wav")
```

Add sherpa-onnx's Android AAR (or its Kotlin API + JNI libraries) to the app, and build the front
end for each ABI with the NDK, e.g.:

```bash
cmake -S mobile/frontend -B build/android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
  -DESPEAK_NG_INCLUDE_DIR=<espeak-ng-1.51>/src/include -DESPEAK_NG_LIBRARY=<libespeak-ng.so for arm64>
cmake --build build/android-arm64 -j
```

What has been tested: the front end and its parity on Linux x86-64, the JNI library from a JVM
(same class and method names as the Kotlin wrapper), and the exported models end to end in
sherpa-onnx 2.0.1 (Python). The Android NDK build and the Kotlin classes have not been run on a device yet.

## Desktop / Python

```bash
pip install africa-tts      # the Python package runs this same sherpa-onnx bundle
africa-tts "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o out.wav
```

## Exporting your own model

`export/onnx/export_onnx.py --model-dir <pocket-tts model folder> --out-dir <dir> --quantize`
exports any pocket-tts model of this family (any layer count) to the files above, with the
model's BOS-before-voice baked into `encoder.onnx` so stock sherpa-onnx can run it.
`test_parity.py`, `test_tokenizer.py` and `test_sherpa.py` check ONNX vs PyTorch, the tokenizer
vs sherpa-onnx's re-implementation, and end-to-end synthesis. Then `mobile/make_bundle.py`.
