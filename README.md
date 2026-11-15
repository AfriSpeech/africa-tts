# Africa TTS

Text-to-speech for **92 African languages and English**, with voice cloning, built on
[Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts). Every language has a default
voice, so all you need is text and a language code.

🎙️ [**Try it live**](https://huggingface.co/spaces/AfriSpeech/africa-tts-demo) ·
🔊 [Samples](https://huggingface.co/spaces/AfriSpeech/africa-tts-samples) ·
🤗 [Model](https://huggingface.co/AfriSpeech/africa-tts)

> An ongoing project: native-speaker feedback is what improves it most. See
> [CONTRIBUTING.md](https://github.com/AfriSpeech/africa-tts/blob/main/CONTRIBUTING.md).

## ⚡ Fast and high-fidelity speech on CPU, no GPU needed

**Africa TTS runs on PyTorch (fp32 unquantised) by default: fast, full-fidelity audio without
quantization degradation, faster than real time on an ordinary CPU.** On a 4-vCPU cloud machine it speaks 2.4×
faster than real time ([speed benchmark](https://github.com/AfriSpeech/africa-tts/blob/main/docs/BENCHMARKS.md#speed)).
It also runs on an NVIDIA GPU with `device="cuda"`.

## 🎯 Intelligible

On held-out sentences transcribed by Omnilingual ASR, Africa TTS is more intelligible than
MMS-TTS in 21 of 27 languages. The 813M OmniVoice, seven times its size, is more intelligible in
24 of 29, by small margins:
[benchmark](https://github.com/AfriSpeech/africa-tts/blob/main/docs/BENCHMARKS.md#intelligibility).

## Install

```bash
pip install africa-tts
```

English also needs espeak-ng (`apt install espeak-ng` / `brew install espeak-ng`). The first time,
it asks before downloading the model (about 440 MB) from Hugging Face, then caches it and works
offline. In scripts, approve the download with `--yes`, `AfricaTTS.load(download=True)`, or
`africa-tts --download` beforehand. On a CPU-only Linux machine, you can install the CPU build of
PyTorch with `pip install --extra-index-url https://download.pytorch.org/whl/cpu africa-tts`.

## Use

```bash
africa-tts "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o twi.wav
africa-tts --list-languages
africa-tts serve                       # the API server below, on port 8000
```

```python
from africa_tts import AfricaTTS

tts = AfricaTTS.load()                                                  # PyTorch, CPU (default)
audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")              # default Twi voice
audio = tts.generate("Hello Ghana!", lang="eng", voice="my_voice.wav")  # clone a 3-10 s clip
tts.save(audio, "out.wav")

# Code-switching: English loanwords & entities are spoken in English (enabled by default)
audio_cs = tts.generate("Mepɛ sɛ mekɔ high school wɔ regional hospital no nkyɛn.", lang="twi")

for chunk in tts.stream("Akwaaba! Yɛfrɛ me Kofi.", lang="twi"):         # 80 ms chunks
    ...

tts_gpu = AfricaTTS.load(device="cuda")                                 # PyTorch on NVIDIA GPU
```

- **Languages:** ISO 639-3 codes (`twi`, `hau`, `yor`, `swh`, `zul`, `amh`, `eng`, ...); see
  `africa-tts --list-languages`.
- **Code-switching:** In multilingual speech, African languages frequently borrow English nouns,
  technical terms, and named entities. Africa TTS automatically detects English words and phrases
  (curated and language-ID-verified from `AfriSpeech/africa-nouns` and `AfriSpeech/african-named-entities`)
  and pronounces them with English phonetics (`espeak-ng`) while preserving the surrounding words
  in native African phonology. Disable with `codeswitch=False`.
- **Numbers:** spell them out ("twenty", not "20").

## Run it elsewhere

| | |
|---|---|
| **API server** | `africa-tts serve`: HTTP, WebSocket streaming, OpenAI-compatible; PyTorch CPU/GPU; Docker or Modal: [server/](https://github.com/AfriSpeech/africa-tts/tree/main/server) |
| **On device** | sherpa-onnx bundles, C front end, Android wrapper: [mobile/](https://github.com/AfriSpeech/africa-tts/tree/main/mobile) |
| **C++ / GGUF (audio.cpp)** | Standalone C++ inference without Python/PyTorch (uses quantized models with some audio quality degradation compared to unquantized PyTorch): [docs/AUDIO_CPP.md](docs/AUDIO_CPP.md) |
| **Experimental languages** | 657 untrained languages, read through their closest trained one; help us test: [docs/EXPERIMENTAL_LANGUAGES.md](https://github.com/AfriSpeech/africa-tts/blob/main/docs/EXPERIMENTAL_LANGUAGES.md) |

## More

- [Benchmarks](https://github.com/AfriSpeech/africa-tts/blob/main/docs/BENCHMARKS.md): speed, and intelligibility per language
- [Model, how it was made, data and limitations](https://github.com/AfriSpeech/africa-tts/blob/main/docs/MODEL.md)
- [Training recipe](https://github.com/AfriSpeech/africa-tts/blob/main/recipes/africa/README.md) · [ONNX export](https://github.com/AfriSpeech/africa-tts/tree/main/export/onnx) · [training code](https://github.com/AfriSpeech/africa-tts/tree/main/training)
- [Status and roadmap](https://github.com/AfriSpeech/africa-tts/blob/main/docs/ROADMAP.md) · [Contributing](https://github.com/AfriSpeech/africa-tts/blob/main/CONTRIBUTING.md)

## License

Code: MIT ([LICENSE](https://github.com/AfriSpeech/africa-tts/blob/main/LICENSE)). Model and default voices: CC-BY-NC-4.0. espeak-ng (English) is
GPL-3.0-or-later. Built by [AfriSpeech](https://huggingface.co/AfriSpeech).
