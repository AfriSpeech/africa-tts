# Model, data and limitations

## How it was made

Africa TTS is a 6-layer, 110M-parameter (codec included) Pocket TTS model. It was distilled from
a 24-layer base model, [Africa TTS Base](https://huggingface.co/AfriSpeech/africa-tts-base),
trained on the IPA of 897 hours of speech. The base model needs classifier-free guidance (2.0) to
sound its best, which runs the backbone twice per frame; the student was trained to match that
guided output, so the guidance is in its weights. Per frame, Africa TTS computes 6 layers where
the guided base computes 48, and it is as intelligible
([benchmark](BENCHMARKS.md#intelligibility)). Recipe: [recipes/africa/](../recipes/africa/README.md).

The base model is released for research and is not supported by the `africa-tts` package.
[recipes/africa/guidance.py](../recipes/africa/guidance.py) runs it with guidance (`load_base()`),
matching the training sampler ([tests/test_guidance_parity.py](../tests/test_guidance_parity.py)).

## Text front end

The model reads **IPA**, not spelling. The same conversion must run everywhere, because the
model only knows the IPA it was trained on:

- **Python** (`africa_tts.to_ipa`): [africa-g2p](https://github.com/AfriSpeech/africa-g2p) pinned to
  commit `76b53a8`. The 0.3.2 release on PyPI predates fixes to the Ghanaian tables and gives
  different IPA for Twi, Fante, Ewe, Igbo and Rundi, so do not substitute it.
- **English:** espeak-ng 1.51 (en-us) plus africa-g2p's normalisation, word by word. espeak-ng
  1.52 changes about 2% of English words.
- **C / mobile** (`mobile/frontend`): africa-g2p's C library at the same commit, and espeak-ng
  1.51. Byte-identical to training on every validation sentence and on 146,578 of 146,580
  English training words; `mobile/frontend/tests/check_parity.py` checks any build.

## Using plain pocket-tts

The model loads in [pocket-tts](https://github.com/kyutai-labs/pocket-tts) directly, given IPA
converted exactly as in training (its guidance is in the weights, so nothing else is needed):

```python
# pip install "pocket-tts>=4.0.1" "africa-g2p[english] @ git+https://github.com/AfriSpeech/africa-g2p@76b53a8"
import re
import scipy.io.wavfile
from africa_g2p import g2p
from pocket_tts import TTSModel

_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)

def to_ipa(text: str, lang: str) -> str:
    """Text -> the IPA the model was trained on. lang: ISO 639-3 code."""
    if lang != "eng":
        return g2p(text, lang, output="ipa")
    words = []  # English (espeak-ng): word by word, keeping punctuation, as in training
    for w in text.split():
        lead, core, trail = _PUNCT.match(w).groups()
        words.append(lead + (g2p(core, "eng", output="ipa").replace(" ", "") if core else "") + trail)
    return " ".join(words)

model = TTSModel.load_model(config="hf://AfriSpeech/africa-tts/config.yaml")
voice = model.get_state_for_audio_prompt("hf://AfriSpeech/africa-tts/voices/twi.wav")
audio = model.generate_audio(voice, to_ipa("Akwaaba! Yɛfrɛ me Kofi.", "twi"))
scipy.io.wavfile.write("out.wav", model.sample_rate, audio.numpy())
```

Or from the command line: `uvx pocket-tts generate --config hf://AfriSpeech/africa-tts/config.yaml
--voice hf://AfriSpeech/africa-tts/voices/twi.wav --text "<IPA>"`. The config turns off Pocket
TTS's first-letter capitalisation and character replacement, since the input is IPA.

## Default voices

One per language, 3.5-6 s of real speech in that language: a female voice from
[africa-female-speech-v2-best60](https://huggingface.co/datasets/AfriSpeech/africa-female-speech-v2-best60)
(highest-rated clip) for the 17 languages it covers (Amharic, Chichewa, Hausa, Igbo,
Kinyarwanda, Rundi, Sepedi, Shona, Swahili, Swati, Tigrinya, Tsonga, Twi, Venda, Xhosa, Yoruba,
Zulu), a JW recording for the others (English: Ghanaian English), at
[AfriSpeech/africa-tts/voices](https://huggingface.co/AfriSpeech/africa-tts/tree/main/voices).

## Training data

897 hours of real speech: JW recordings (AfriSpeech Selector, 383 h), africa-female-speech-v2
(192 h), Ghanaian English (156 h) and HiFiTTS-2 (166 h). Clips with numbers in the transcript
were removed; word timings come from ctc-forced-aligner for every language. Languages and hours:
[africa_tts/languages.json](../africa_tts/languages.json). The full recipe:
[recipes/africa/README.md](../recipes/africa/README.md).

## Limitations

- Mostly read religious speech for the African languages; news, conversation and technical
  vocabulary are less natural, and languages with under ~2 h of data are weak.
- Numbers must be spelled out: the model never saw digits.
- Some africa-g2p tables do not mark tone (e.g. Fon, Ewe, Igbo, Ga), and a few rare Ge'ez
  syllables (e.g. ኳ) are missing from the Amharic table.
- Loanwords follow the host language's spelling rules.
- Voice cloning can imitate real people: do not use it to impersonate anyone or without consent.
