# Africa TTS

Text-to-speech for **92 African languages and English**, with voice cloning, built on
[Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts). One model reads all languages
as IPA produced by [africa-g2p](https://github.com/AfriSpeech/africa-g2p), and every language
ships with a default voice, so you only need text and a language code.

🔊 [Samples in 53 languages](https://huggingface.co/spaces/AfriSpeech/africa-tts-samples) ·
🤗 [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts) (24 layers) ·
🤗 [Africa TTS Turbo](https://huggingface.co/AfriSpeech/africa-tts-turbo) (6 layers, distilled, fast on CPU)

This repository has everything: the inference package, the full training recipe (data,
alignment, IPA, tokenizer, teacher, distillation, release), ONNX export, and on-device
deployment with sherpa-onnx.

> **An ongoing project.** This is the first release, not a finished product: many languages
> are thin, some are missing, and native-speaker feedback is what improves them most. See the
> [roadmap](#roadmap) and [CONTRIBUTING.md](CONTRIBUTING.md): listening reports, G2P fixes,
> data, evaluation and mobile work are all welcome.

## Quick start

```bash
pip install "africa-tts @ git+https://github.com/AfriSpeech/africa-tts"
# English also needs espeak-ng: apt install espeak-ng / brew install espeak-ng
africa-tts "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o twi.wav
africa-tts "Sannu da zuwa." --lang hau --model africa-tts -o hausa.wav
africa-tts --list-languages
```

```python
from africa_tts import AfricaTTS

tts = AfricaTTS.load("africa-tts-turbo")        # or "africa-tts" (see "Three ways to run it")
audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")              # default Twi voice
audio = tts.generate("Hello Ghana!", lang="eng", voice="my_voice.wav")  # clone a voice
tts.save(audio, "out.wav")
```

- **Languages:** ISO 639-3 codes, e.g. `twi fat gaa ewe hau yor ibo swh zul xho amh kin sna lin
  fon eng`; the full list with training hours is in
  [africa_tts/languages.json](africa_tts/languages.json).
- **Default voices:** one per language, 3.5-6 s of real speech in that language: a female
  voice from [africa-female-speech-v2-best60](https://huggingface.co/datasets/AfriSpeech/africa-female-speech-v2-best60)
  (highest-rated clip) for the 17 languages it covers (Amharic, Chichewa, Hausa, Igbo,
  Kinyarwanda, Rundi, Sepedi, Shona, Swahili, Swati, Tigrinya, Tsonga, Twi, Venda, Xhosa, Yoruba,
  Zulu), a JW recording for the others (English: Ghanaian English), at
  [AfriSpeech/africa-tts/voices](https://huggingface.co/AfriSpeech/africa-tts/tree/main/voices).
  Pass `voice=` to clone any 3-10 s clip instead.
- **Numbers:** spell them out. Training dropped every clip with a numeral, so the model has
  never read digits.

## 🧪 Experimental: 657 more languages, help us test

The model reads IPA, so it can be tried on **any language africa-g2p has an IPA table for**:
657 African languages it was never trained on, each paired with its closest trained language
(Bambara with Jula, Dagbani with Dagaare, Tafi with Ewe, ...), whose voice it borrows.

```bash
africa-tts --list-experimental
africa-tts "Your sentence." --lang bam -o test.wav
```

Native speakers: tell us how it sounds with an
[experimental-language report](https://github.com/AfriSpeech/africa-tts/issues/new?template=experimental-language.yml).
The list, tiers and how they were chosen: [docs/EXPERIMENTAL_LANGUAGES.md](docs/EXPERIMENTAL_LANGUAGES.md).

## Three ways to run it

| Mode | How it works | Compute per audio frame | Runs in |
|---|---|---|---|
| **Africa TTS, guided** | 24-layer model with classifier-free guidance 2.0: every frame runs the backbone twice (with and without the text and voice) and pushes the output towards the conditioned one | 2 × 24 layers | Python (`AfricaTTS.load("africa-tts")`) |
| **Africa TTS, unguided** | the same 24-layer model, one pass per frame | 24 layers | Python (`AfricaTTS.load("africa-tts", cfg=1.0)`), sherpa-onnx (on device) |
| **Africa TTS Turbo** | 6-layer model distilled from the guided 24-layer model, so guidance is built into its weights; one pass per frame | 6 layers | Python (`AfricaTTS.load()`, **the package default**), sherpa-onnx (on device) |

The package loads **Turbo** unless you name a model (`AfricaTTS.load()`, or `africa-tts` without
`--model`); loading `africa-tts` uses guidance 2.0 unless you pass `cfg=1.0`.

The modes trade compute for different things, not along one quality scale: which sounds best
can differ by language and listener, and per-language evaluations are on the
[roadmap](#roadmap). Listen on the [samples page](https://huggingface.co/spaces/AfriSpeech/africa-tts-samples)
and try them on your own text.

Guidance is implemented in [africa_tts/guidance.py](africa_tts/guidance.py) on top of
pocket-tts's own generation, and matches the training code's guided sampler to float precision
([tests/test_guidance_parity.py](tests/test_guidance_parity.py)). sherpa-onnx cannot run
guidance (it needs the second pass), so on device the choice is the unguided 24-layer model or
Turbo.

## Repository layout

| Path | What |
|---|---|
| [africa_tts/](africa_tts/) | Python package: `AfricaTTS`, `to_ipa`, CLI, default voices |
| [recipes/africa/](recipes/africa/) | how the models were made: data, alignment, IPA, tokenizer, release |
| [training/](training/) | Pocket TTS training code (Kyutai's, with keep-best checkpoints, early stopping, `--romanize` alignment, tokenizer growth) and the Africa configs |
| [export/onnx/](export/onnx/) | ONNX export for sherpa-onnx, with parity tests |
| [mobile/](mobile/) | on-device deployment (unguided Africa TTS and Turbo): front end in C (africa-g2p + espeak-ng), Android/Kotlin wrapper, sherpa-onnx bundles |

## How the text front end stays consistent

The model only knows the IPA it was trained on, so the same conversion must run everywhere:

- **Python** (`africa_tts.to_ipa`): africa-g2p pinned to commit `76b53a8`. The 0.3.2 release on
  PyPI predates fixes to the Ghanaian tables and gives different IPA for Twi, Fante, Ewe, Igbo
  and Rundi, so do not substitute it.
- **C / mobile** (`mobile/frontend`): africa-g2p's C library at the same commit, and espeak-ng
  1.51 for English. Byte-identical to training on every validation sentence and on 146,578 of
  146,580 English training words; `mobile/frontend/tests/check_parity.py` checks any build.

## Training data

897 hours of real speech: JW recordings (AfriSpeech Selector, 383 h), africa-female-speech-v2
(192 h), Ghanaian English (156 h) and HiFiTTS-2 (166 h). Clips with numbers in the transcript
were removed; word timings come from ctc-forced-aligner for every language. Details:
[recipes/africa/README.md](recipes/africa/README.md).

## Limitations

- Mostly read religious speech for the African languages; news, conversation and technical
  vocabulary are less natural, and languages with under ~2 h of data are weak.
- Some africa-g2p tables do not mark tone (e.g. Fon, Ewe, Igbo, Ga), and a few rare Ge'ez
  syllables (e.g. ኳ) are missing from the Amharic table.
- Loanwords follow the host language's spelling rules.
- Voice cloning can imitate real people: do not use it to impersonate anyone or without consent.

## Status

| Component | State |
|---|---|
| Africa TTS (24 layers) | released, v0.1 (step 100k) |
| Africa TTS Turbo (6 layers) | distillation in progress (run to convergence, early stopping) |
| Python package and CLI | released |
| Guided inference for the teacher (`africa_tts`, cfg 2.0) | released, parity-tested against the training sampler |
| sherpa-onnx export and bundles | released, tested end to end in sherpa-onnx 2.0.1 (Africa TTS unguided; Turbo when released) |
| C front end (africa-g2p + espeak-ng) | released, parity-tested on Linux, JNI tested on a desktop JVM |
| Android | Kotlin wrapper written; NDK build and on-device test still to do |
| iOS | not started |

## Roadmap

Help is welcome on any of these; see [CONTRIBUTING.md](CONTRIBUTING.md).

- **Experimental languages:** collect native-speaker reports on the 657 untrained languages;
  promote the ones that work, and prioritise data for the ones that nearly do.
- **Evaluation:** per-language intelligibility (ASR CER), speaker similarity and native-speaker
  listening tests, published per language so weak languages are visible.
- **Numbers:** spell out numerals per language before G2P (the models never saw digits).
- **Tone:** tone-marked G2P tables where they are missing (Fon, Ewe, Igbo, Ga, ...), then retrain.
- **More data, more languages:** grow the languages under 2 hours, add missing ones (e.g.
  Dagbani validation data), and add non-religious and conversational speech.
- **Mobile:** an Android example app and NDK build of the front end with espeak-ng 1.51; iOS.
- **Packaging:** africa-g2p and africa-tts releases on PyPI, so no git pins are needed.
- **Next training run:** fold in G2P fixes, new data and evaluation results, then distill again.

## License

Code: MIT (see [LICENSE](LICENSE); includes Kyutai's Pocket TTS training code and an ONNX
exporter based on KevinAHM's and csukuangfj's, all MIT). Models and default voices:
CC-BY-NC-4.0, following the terms of the JW recordings in the training data. espeak-ng, used for
English, is GPL-3.0-or-later.

Built by [AfriSpeech](https://huggingface.co/AfriSpeech) / Ghana NLP on top of Kyutai's Pocket TTS.
