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

tts = AfricaTTS.load("africa-tts-turbo")        # or "africa-tts"
audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")              # default Twi voice
audio = tts.generate("Hello Ghana!", lang="eng", voice="my_voice.wav")  # clone a voice
tts.save(audio, "out.wav")
```

- **Languages:** ISO 639-3 codes, e.g. `twi fat gaa ewe hau yor ibo swh zul xho amh kin sna lin
  fon eng`; the full list with training hours is in
  [africa_tts/languages.json](africa_tts/languages.json).
- **Default voices:** one per language, a few seconds of a JW recording in that language
  (English: Ghanaian English; Tigrinya: africa-female-speech-v2), at
  [AfriSpeech/africa-tts/voices](https://huggingface.co/AfriSpeech/africa-tts/tree/main/voices).
  Pass `voice=` to clone any 3-10 s clip instead.
- **Numbers:** spell them out. Training dropped every clip with a numeral, so the model has
  never read digits.

## Which model?

| | Africa TTS | Africa TTS Turbo |
|---|---|---|
| FlowLM layers | 24 (316M parameters) | 6 |
| Speed | slower; GPU or a strong CPU | about 4x less compute, real time on CPU |
| Guidance | none at inference (pocket-tts and sherpa-onnx do not apply CFG) | cfg 2.0 baked in by distillation |

## Repository layout

| Path | What |
|---|---|
| [africa_tts/](africa_tts/) | Python package: `AfricaTTS`, `to_ipa`, CLI, default voices |
| [recipes/africa/](recipes/africa/) | how the models were made: data, alignment, IPA, tokenizer, release |
| [training/](training/) | Pocket TTS training code (Kyutai's, with keep-best checkpoints, early stopping, `--romanize` alignment, tokenizer growth) and the Africa configs |
| [export/onnx/](export/onnx/) | ONNX export for sherpa-onnx, with parity tests |
| [mobile/](mobile/) | on-device front end in C (africa-g2p + espeak-ng), Android/Kotlin wrapper, sherpa-onnx bundles |

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

## License

Code: MIT (see [LICENSE](LICENSE); includes Kyutai's Pocket TTS training code and an ONNX
exporter based on KevinAHM's and csukuangfj's, all MIT). Models and default voices:
CC-BY-NC-4.0, following the terms of the JW recordings in the training data. espeak-ng, used for
English, is GPL-3.0-or-later.

Built by [AfriSpeech](https://huggingface.co/AfriSpeech) / Ghana NLP on top of Kyutai's Pocket TTS.
