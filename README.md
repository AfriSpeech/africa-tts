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

**Africa TTS runs on [audio.cpp](https://github.com/0xShug0/audio.cpp) by default: a 144 MB model,
no PyTorch, faster than real time on an ordinary laptop, streaming its first audio about 0.1 s
after it gets the text.** On a 2020 laptop CPU (AMD Ryzen 5 4500U, 4 threads) it speaks 2.3-2.8×
faster than real time. It has 110M parameters, so it runs on laptops, servers and phones.

The unquantised PyTorch engine is optional (`pip install "africa-tts[torch]"`): it also runs on a
GPU, and on a 4-vCPU cloud machine it speaks 2.4× faster than real time
([speed benchmark](https://github.com/AfriSpeech/africa-tts/blob/main/docs/BENCHMARKS.md#speed)).

## 🎯 Intelligible

On held-out sentences in 47 languages, transcribed by Omnilingual ASR, Africa TTS has a lower
character error rate than MMS-TTS (9.0% vs 10.3%, better in 21 of 27 languages) and the 3.3B
Sunbird Orpheus (5.9% vs 10.3%), and is close to the 813M OmniVoice (8.8% vs 8.4%) at a
seventh of its size: [benchmark](https://github.com/AfriSpeech/africa-tts/blob/main/docs/BENCHMARKS.md#intelligibility).

## Install

```bash
pip install africa-tts                 # default: audio.cpp on the CPU, no PyTorch
pip install "africa-tts[torch]"        # optional: the unquantised PyTorch engine (CPU or GPU)
```

English also needs espeak-ng (`apt install espeak-ng` / `brew install espeak-ng`). The first time,
it asks before downloading the model (about 150 MB; 440 MB for PyTorch) from Hugging Face, then
caches it and works offline. In scripts, approve the download with `--yes`,
`AfricaTTS.load(download=True)`, or `africa-tts --download` beforehand. For PyTorch on a CPU-only
Linux machine, add `--extra-index-url https://download.pytorch.org/whl/cpu` to get the small
CPU build.

## Use

```bash
africa-tts "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o twi.wav
africa-tts --list-languages
africa-tts serve                       # the API server below, on port 8000
```

```python
from africa_tts import AfricaTTS

tts = AfricaTTS.load()                                                  # audio.cpp, CPU
audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")              # default Twi voice
audio = tts.generate("Hello Ghana!", lang="eng", voice="my_voice.wav")  # clone a 3-10 s clip
tts.save(audio, "out.wav")

for chunk in tts.stream("Akwaaba! Yɛfrɛ me Kofi.", lang="twi"):         # 80 ms chunks
    ...

tts = AfricaTTS.load(backend="torch")   # PyTorch, unquantised (pip install "africa-tts[torch]")
tts = AfricaTTS.load(device="cuda")     # PyTorch on an NVIDIA GPU
```

- **Languages:** ISO 639-3 codes (`twi`, `hau`, `yor`, `swh`, `zul`, `amh`, `eng`, ...); see
  `africa-tts --list-languages`.
- **Numbers:** spell them out ("twenty", not "20").

## Run it elsewhere

| | |
|---|---|
| **API server** | `africa-tts serve`: HTTP, WebSocket streaming, OpenAI-compatible; audio.cpp on CPU, or PyTorch (CPU/GPU) with the torch extra; Docker or Modal: [server/](https://github.com/AfriSpeech/africa-tts/tree/main/server) |
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
