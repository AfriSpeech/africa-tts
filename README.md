# Africa TTS

Text-to-speech for **92 African languages and English**, with voice cloning, built on
[Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts). Every language has a default
voice, so all you need is text and a language code.

🎙️ [**Try it live**](https://huggingface.co/spaces/AfriSpeech/africa-tts-demo) ·
🔊 [Samples](https://huggingface.co/spaces/AfriSpeech/africa-tts-samples) ·
🤗 [Africa TTS Turbo](https://huggingface.co/AfriSpeech/africa-tts-turbo) ·
🤗 [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts)

> An ongoing project: native-speaker feedback is what improves it most. See
> [CONTRIBUTING.md](CONTRIBUTING.md).

## ⚡ Runs on a CPU, no GPU needed

**Africa TTS Turbo speaks 2.4× faster than real time on 4 CPU cores, and starts streaming
audio 0.17 s after it gets the text.** A 10-second sentence takes about 4 seconds to generate,
and playback can start almost at once.

| On a 4-core CPU | Africa TTS Turbo | Africa TTS (24 layers) |
|---|---|---|
| Speed | **2.4-2.5× real time** | 0.8-0.9× unguided, 0.5× guided |
| Time to first audio | **0.17 s** | 0.5-0.7 s |
| Size | 6 layers; 209 MB on-device bundle (int8) | 24 layers; 437 MB bundle |
| Use it for | laptops, servers, phones, real-time apps | offline work, or a GPU |

Measured on a dedicated 4-vCPU cloud machine (Modal), PyTorch on one thread, sherpa-onnx int8 on
four; your laptop may be faster. Measure your own machine with
[`benchmarks/speed.py`](benchmarks/speed.py); details in [docs/BENCHMARKS.md](docs/BENCHMARKS.md#speed).

## Install

```bash
pip install "africa-tts @ git+https://github.com/AfriSpeech/africa-tts"
# English also needs espeak-ng: apt install espeak-ng / brew install espeak-ng
```

## Use

```bash
africa-tts "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o twi.wav
africa-tts --list-languages
```

```python
from africa_tts import AfricaTTS

tts = AfricaTTS.load()                                                  # Africa TTS Turbo
audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")              # default Twi voice
audio = tts.generate("Hello Ghana!", lang="eng", voice="my_voice.wav")  # clone a 3-10 s clip
tts.save(audio, "out.wav")
```

- **Languages:** ISO 639-3 codes (`twi`, `hau`, `yor`, `swh`, `zul`, `amh`, `eng`, ...); see
  `africa-tts --list-languages`.
- **Numbers:** spell them out ("twenty", not "20").
- **Models:** `AfricaTTS.load()` gives **Turbo** (6 layers, fast on CPU). For the 24-layer
  model use `AfricaTTS.load("africa-tts")`, guided by default, or `cfg=1.0` for a single pass:
  [how the three modes work](docs/MODES.md).

## Run it elsewhere

| | |
|---|---|
| **API server** | HTTP, WebSocket streaming, OpenAI-compatible; Docker or Modal, CPU only: [server/](server/) |
| **On device** | sherpa-onnx bundles, C front end, Android wrapper: [mobile/](mobile/) |
| **Experimental languages** | 657 untrained languages, read through their closest trained one; help us test: [docs/EXPERIMENTAL_LANGUAGES.md](docs/EXPERIMENTAL_LANGUAGES.md) |

## More

- [Benchmark](docs/BENCHMARKS.md): intelligibility (ASR CER) per language, against MMS-TTS and OmniVoice
- [Model, data and limitations](docs/MODEL.md)
- [Training recipe](recipes/africa/README.md) · [ONNX export](export/onnx/) · [training code](training/)
- [Status and roadmap](docs/ROADMAP.md) · [Contributing](CONTRIBUTING.md)

## License

Code: MIT ([LICENSE](LICENSE)). Models and default voices: CC-BY-NC-4.0. espeak-ng (English) is
GPL-3.0-or-later. Built by [AfriSpeech](https://huggingface.co/AfriSpeech) / Ghana NLP.
