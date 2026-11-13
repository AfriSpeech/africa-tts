# Deploying Africa TTS with audio.cpp (C++ / GGUF)

> **Mainstream Engine vs. Quantized GGUF:**
> - **Mainstream:** Africa TTS uses **PyTorch (fp32 unquantised)** on CPU or GPU as its default, production-grade engine (`pip install africa-tts`). PyTorch generates audio at full floating-point precision, ensuring maximum vocal fidelity, smooth intonation, and absence of quantization artifacts, running at ~2.4× real-time on standard CPUs.
> - **audio.cpp (GGUF):** This guide is for developers who require a standalone C++ binary without a Python runtime or PyTorch dependencies, using [audio.cpp](https://github.com/0xShug0/audio.cpp).
> - ⚠️ **Quality Note:** The audio.cpp path runs **quantized GGUF models** (e.g. `q8_0`). While quantization reduces disk footprint (~144 MB) and memory usage, it introduces **noticeable degradation in audio quality and naturalness** compared to the mainstream unquantized fp32 PyTorch engine. If speech quality and fidelity are primary goals, use the mainstream PyTorch engine.

---

## 1. Overview

[audio.cpp](https://github.com/0xShug0/audio.cpp) is a pure C++ audio inference library powered by `ggml`. Africa TTS is compatible with audio.cpp's Pocket TTS implementation via quantized GGUF models.

- **Model size:** ~144 MB (q8_0 quantized 110M model)
- **Quality Trade-off:** Quantized weights trade off audio fidelity for minimal memory footprint and zero Python dependencies.
- **Platform support:** Linux, macOS (Apple Silicon & Intel), Windows

---

## 2. Download Model and Voices

Pre-quantized 110M GGUF models and bundled voices are hosted on Hugging Face:

```bash
# Model repository: https://huggingface.co/AfriSpeech/africa-tts
# Download the 110M Q8_0 GGUF model:
wget https://huggingface.co/AfriSpeech/africa-tts/resolve/main/gguf/africa-tts-q8_0.gguf

# Download reference voice clips (3-10 s clean speech):
wget https://huggingface.co/AfriSpeech/africa-tts/resolve/main/voices/twi.wav
wget https://huggingface.co/AfriSpeech/africa-tts/resolve/main/voices/yor.wav
wget https://huggingface.co/AfriSpeech/africa-tts/resolve/main/voices/hau.wav
```

---

## 3. Build audio.cpp

Clone and build the `audio.cpp` CLI tools with `cmake`:

```bash
git clone https://github.com/0xShug0/audio.cpp.git
cd audio.cpp

# Configure and compile the TTS CLI tool
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target audiocpp_tts
```

The compiled binary will be located at `build/bin/audiocpp_tts`.

---

## 4. Converting Text to IPA

Africa TTS models read International Phonetic Alphabet (IPA) tokens. Use [africa-g2p](https://github.com/AfriSpeech/africa-g2p) (available in Python and native C) to convert text to IPA:

### Python CLI:
```bash
pip install africa-g2p
africa-g2p "Akwaaba! Yɛfrɛ me Kofi." twi
# Output: akʷaaba! jɛfɾɛ me kofi.
```

### Native C CLI:
```bash
# Build native afrig2p CLI from https://github.com/AfriSpeech/africa-g2p/tree/main/native
echo "Akwaaba! Yɛfrɛ me Kofi." | ./build/afrig2p -d data/twi.afg2p -m ipa
```

---

## 5. Synthesize Speech via audio.cpp CLI

Pass the GGUF model, reference voice WAV, and phonemized IPA text to `audiocpp_tts`:

```bash
./build/bin/audiocpp_tts \
    --model africa-tts-q8_0.gguf \
    --voice twi.wav \
    --text "akʷaaba! jɛfɾɛ me kofi." \
    --temperature 0.3 \
    --output output.wav
```

Output: 24 kHz mono 16-bit PCM WAV.

---

## 6. C / C++ API Integration

To embed Africa TTS into a C/C++ application, link against `libaudiocpp` and use its session API:

```c
#include <stdio.h>
#include <stdlib.h>
#include "audiocpp.h"

int main() {
    audiocpp_registry_t registry;
    audiocpp_model_t model;
    audiocpp_session_t session;

    // 1. Initialize registry and load GGUF model
    audiocpp_registry_create(NULL, &registry);
    audiocpp_model_config_t mcfg = { .threads = 4 };
    audiocpp_model_load(registry, "africa-tts-q8_0.gguf", &mcfg, NULL, &model);

    // 2. Create TTS session
    audiocpp_backend_config_t bcfg = { .threads = 4 };
    audiocpp_session_create(model, "tts", "offline", &bcfg, NULL, &session);

    // 3. Build synthesis request
    audiocpp_request_t req = audiocpp_request_create();
    audiocpp_request_set_text(req, "akʷaaba! jɛfɾɛ me kofi.", NULL);
    audiocpp_request_set_option(req, "temperature", "0.3");

    // Load reference voice audio (float32 array, 24 kHz mono)
    // audiocpp_request_set_voice_audio(req, voice_samples, num_samples, 24000, 1);

    // 4. Run synthesis
    audiocpp_result_t result;
    audiocpp_session_run(session, req, &result);

    // 5. Retrieve audio samples
    const float *samples;
    size_t n_samples;
    int sr, channels;
    audiocpp_result_audio(result, &samples, &n_samples, &sr, &channels);
    printf("Synthesized %zu samples at %d Hz\n", n_samples, sr);

    // Cleanup
    audiocpp_result_free(result);
    audiocpp_request_free(req);
    audiocpp_session_free(session);
    audiocpp_model_free(model);
    audiocpp_registry_free(registry);
    return 0;
}
```

---

## 7. Converting Custom Checkpoints to GGUF

If you train or fine-tune your own Africa TTS checkpoint, convert `model.safetensors` to GGUF using `audiocpp_gguf`:

```bash
# In audio.cpp build directory:
cmake --build build -j --target audiocpp_gguf

./build/bin/audiocpp_gguf \
    --input /path/to/model.safetensors \
    --root /path/to/model_directory \
    --output africa-tts-q8_0.gguf \
    --type q8_0 \
    --family pocket_tts
```

`model_directory` must contain `config.yaml`, `model.safetensors`, and the SentencePiece `tokenizer.model`.
