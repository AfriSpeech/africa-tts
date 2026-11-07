# Africa TTS

Text-to-speech for **92 African languages and English**, with voice cloning, built on
[Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts). Every language has a default
voice, so all you need is text and a language code.

🎙️ [**Try it live**](https://huggingface.co/spaces/AfriSpeech/africa-tts-demo) ·
🔊 [Samples](https://huggingface.co/spaces/AfriSpeech/africa-tts-samples) ·
🤗 [Model](https://huggingface.co/AfriSpeech/africa-tts)

> An ongoing project: native-speaker feedback is what improves it most. See
> [CONTRIBUTING.md](https://github.com/AfriSpeech/africa-tts/blob/main/CONTRIBUTING.md).

## ⚡ Runs on a CPU, no GPU needed

**Africa TTS speaks 2.4× faster than real time on 4 CPU cores, and starts streaming audio
0.17 s after it gets the text.** A 10-second sentence takes about 4 seconds to generate, and
playback can start almost at once. It has 110M parameters, and the on-device bundle (int8) is
209 MB, so it runs on laptops, servers and phones.

Measured on a dedicated 4-vCPU cloud machine (Modal); your laptop may be faster. Measure your own
machine with [`benchmarks/speed.py`](https://github.com/AfriSpeech/africa-tts/blob/main/benchmarks/speed.py); details in
[docs/BENCHMARKS.md](https://github.com/AfriSpeech/africa-tts/blob/main/docs/BENCHMARKS.md#speed).

## 🎯 Intelligible

On held-out sentences in 47 languages, transcribed by Omnilingual ASR, Africa TTS has a lower
character error rate than MMS-TTS (9.0% vs 10.3%, better in 21 of 27 languages) and the 3.3B
Sunbird Orpheus (5.9% vs 10.3%), and is close to the 813M OmniVoice (8.8% vs 8.4%) at a
seventh of its size: [benchmark](https://github.com/AfriSpeech/africa-tts/blob/main/docs/BENCHMARKS.md#intelligibility).

## Install

```bash
# CPU (laptops, servers): small CPU-only PyTorch
pip install africa-tts --extra-index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (Linux): PyPI's PyTorch is the CUDA build
pip install africa-tts
```

English also needs espeak-ng (`apt install espeak-ng` / `brew install espeak-ng`). On macOS and
Windows, plain `pip install africa-tts` gives the CPU build. The first time, it asks before
downloading the model (about 440 MB) from Hugging Face, then caches it and works offline. In
scripts, approve the download with `--yes`, `AfricaTTS.load(download=True)`, or
`africa-tts --download` beforehand.

## Use

```bash
africa-tts "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o twi.wav
africa-tts --list-languages
```

```python
from africa_tts import AfricaTTS

tts = AfricaTTS.load()                                  # or AfricaTTS.load(device="cuda") on a GPU
audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")              # default Twi voice
audio = tts.generate("Hello Ghana!", lang="eng", voice="my_voice.wav")  # clone a 3-10 s clip
tts.save(audio, "out.wav")
```

- **Languages:** ISO 639-3 codes (`twi`, `hau`, `yor`, `swh`, `zul`, `amh`, `eng`, ...); see
  `africa-tts --list-languages`.
- **Numbers:** spell them out ("twenty", not "20").

## Run it elsewhere

| | |
|---|---|
| **API server** | HTTP, WebSocket streaming, OpenAI-compatible; Docker or Modal, CPU only: [server/](https://github.com/AfriSpeech/africa-tts/tree/main/server) |
| **On device** | sherpa-onnx bundles, C front end, Android wrapper: [mobile/](https://github.com/AfriSpeech/africa-tts/tree/main/mobile) |
| **Experimental languages** | 657 untrained languages, read through their closest trained one; help us test: [docs/EXPERIMENTAL_LANGUAGES.md](https://github.com/AfriSpeech/africa-tts/blob/main/docs/EXPERIMENTAL_LANGUAGES.md) |

## More

- [Benchmarks](https://github.com/AfriSpeech/africa-tts/blob/main/docs/BENCHMARKS.md): speed, and intelligibility per language
- [Model, how it was made, data and limitations](https://github.com/AfriSpeech/africa-tts/blob/main/docs/MODEL.md)
- [Training recipe](https://github.com/AfriSpeech/africa-tts/blob/main/recipes/africa/README.md) · [ONNX export](https://github.com/AfriSpeech/africa-tts/tree/main/export/onnx) · [training code](https://github.com/AfriSpeech/africa-tts/tree/main/training)
- [Status and roadmap](https://github.com/AfriSpeech/africa-tts/blob/main/docs/ROADMAP.md) · [Contributing](https://github.com/AfriSpeech/africa-tts/blob/main/CONTRIBUTING.md)

## License

Code: MIT ([LICENSE](https://github.com/AfriSpeech/africa-tts/blob/main/LICENSE)). Model and default voices: CC-BY-NC-4.0. espeak-ng (English) is
GPL-3.0-or-later. Built by [AfriSpeech](https://huggingface.co/AfriSpeech) / Ghana NLP.
