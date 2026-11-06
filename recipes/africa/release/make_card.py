"""Write the HF model card (README.md) for africa-tts or africa-tts-turbo.

Usage: python make_card.py {teacher|turbo} LANG_HOURS.json UDHR_META.json OUT/README.md
"""

import json
import sys

SPACE = "https://huggingface.co/spaces/AfriSpeech/africa-tts-samples"
NICE = {"english_eng": "English", "swahili_swa": "Swahili", "twi_twi": "Twi", "yoruba_yor": "Yoruba",
        "xhosa_xho": "Xhosa"}


# Hub language tags: every training language's current ISO 639-3 code, plus its ISO 639-1 code
# and macrolanguage where they exist (the Hub's language filter mostly uses 639-1, e.g. "sw").
RETIRED = {"daf": "dnj"}  # Dan (Yacouba, Côte d'Ivoire): ISO split daf into dnj + lda in 2013
MACRO = {"swh": ["swa", "sw"], "twi": ["aka", "ak"], "fat": ["aka", "ak"]}


def hub_languages(codes: list[str]) -> list[str]:
    import pycountry

    out: list[str] = []
    for c in codes:
        c = RETIRED.get(c, c)
        lang = pycountry.languages.get(alpha_3=c)
        assert lang is not None, f"not a current ISO 639-3 code: {c}"
        for t in [getattr(lang, "alpha_2", None), c, *MACRO.get(c, [])]:
            if t and t not in out:
                out.append(t)
    return out + ["multilingual"]


def nice(n: str) -> str:
    if n in NICE:
        return NICE[n]
    return n.split("_")[0].capitalize() if "_" in n else n


def main() -> None:
    kind, hours_p, meta_p, out = sys.argv[1:5]
    hours = json.load(open(hours_p))
    meta = json.load(open(meta_p))
    udhr = {it["g2p"] for it in meta["items"]}
    turbo = kind == "turbo"
    repo = "AfriSpeech/africa-tts-turbo" if turbo else "AfriSpeech/africa-tts"
    langs = hours["langs"]
    african = [x for x in langs if x[0] != "eng"]
    src_tot: dict[str, float] = {}
    for _, _, _, src in langs:
        for k, v in src.items():
            src_tot[k] = src_tot.get(k, 0) + v
    rows = "\n".join(
        f"| {nice(n)} | `{g}`{f' (ISO: `{RETIRED[g]}`)' if g in RETIRED else ''} | {h:.1f} | "
        f"{'✓' if g in udhr else ''} |" for g, n, h, _ in
        sorted(langs, key=lambda x: (x[0] != "eng", -x[2])))

    if turbo:
        what = (f"**Africa TTS Turbo** is the fast, 6-layer version of [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts): "
                f"a [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) text-to-speech model for "
                f"**{len(african)} African languages and English**, with voice cloning. It was distilled from the "
                f"24-layer Africa TTS teacher, has classifier-free guidance built in, and is sized to run faster "
                f"than real time on a CPU.")
        arch = ("6-layer FlowLM (d_model 1024), distilled from the 24-layer teacher (`checkpoint 100k`). "
                "Text table, flow head, EOS head and latent statistics copied from the teacher; the 6 layers "
                "were seeded from teacher layers 0, 1, 2, 21, 22, 23 and trained to match the teacher's "
                "guided (cfg 2.0) outputs (Kyutai's `depth_distill` recipe, lr 4e-4 cosine, batch 64). "
                "Guidance is baked in: generate with cfg 1 (the default).")
    else:
        what = (f"**Africa TTS** is a [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) text-to-speech "
                f"model for **{len(african)} African languages and English**, with voice cloning from a few "
                f"seconds of audio. Text is read as IPA, produced by "
                f"[africa-g2p](https://github.com/AfriSpeech/africa-g2p). This is the 24-layer model; "
                f"[Africa TTS Turbo](https://huggingface.co/AfriSpeech/africa-tts-turbo) is its fast 6-layer "
                f"distilled version.")
        arch = ("24-layer FlowLM (d_model 1024, 316M parameters) + Mimi codec (24 kHz, 12.5 Hz frames). "
                "Kyutai's new-language recipe (`finetune_language`): warm start from Kyutai's English 24-layer "
                "model (`english_2026-04_24l`) with a fresh text table for a new 4,000-token IPA SentencePiece "
                "(Unigram) tokenizer; nothing frozen; lr 2e-4 constant after 1k warmup; batch 64; LSD flow; "
                "text and voice dropout 0.2; EMA 0.999. Validated every 1k steps, early-stopped at 113k "
                "(best validation loss at 93k). The released weights are the **100k** checkpoint (EMA), "
                "chosen by listening.")

    card = f"""---
license: cc-by-nc-4.0
language:
{chr(10).join(f"- {t}" for t in hub_languages([g for g, *_ in langs]))}
pipeline_tag: text-to-speech
library_name: pocket-tts
base_model: kyutai/pocket-tts
tags:
- text-to-speech
- african-languages
- ipa
- voice-cloning
- pocket-tts
---

# {"Africa TTS Turbo" if turbo else "Africa TTS"}

{what}

🔊 **Listen:** [samples in every language]({SPACE}) — Articles 1, 3 and 5 of the Universal
Declaration of Human Rights, read in {len(udhr)} languages.

## Quick start

```bash
pip install "pocket-tts>=4.0.1" \\
  "africa-g2p[english] @ git+https://github.com/AfriSpeech/africa-g2p@76b53a8"
```

The model reads **IPA**, not spelling. Convert text with africa-g2p exactly as in training.
Use the pinned africa-g2p commit above: the 0.3.2 release on PyPI predates fixes to the Ghanaian
tables and gives different IPA for Twi, Fante, Ewe, Igbo and Rundi. English needs
[espeak-ng](https://github.com/espeak-ng/espeak-ng) installed (`apt install espeak-ng`,
`brew install espeak-ng`).

```python
import re
import scipy.io.wavfile
from africa_g2p import g2p
from pocket_tts import TTSModel

_PUNCT = re.compile(r"^(\\W*)(.*?)(\\W*)$", re.S)

def to_ipa(text: str, lang: str) -> str:
    \"\"\"Text -> the IPA the model was trained on. lang: ISO 639-3 code (see the table below).\"\"\"
    if lang != "eng":
        return g2p(text, lang, output="ipa")
    words = []  # English (espeak-ng): word by word, keeping punctuation, as in training
    for w in text.split():
        lead, core, trail = _PUNCT.match(w).groups()
        words.append(lead + (g2p(core, "eng", output="ipa").replace(" ", "") if core else "") + trail)
    return " ".join(words)

model = TTSModel.load_model(config="hf://{repo}/config.yaml")
voice = model.get_state_for_audio_prompt("speaker.wav")  # 3-10 s of the voice to clone
audio = model.generate_audio(voice, to_ipa("Akwaaba! Yɛfrɛ me Kofi.", "twi"))
scipy.io.wavfile.write("out.wav", model.sample_rate, audio.numpy())
```

CLI (`--text` must already be IPA):

```bash
pocket-tts generate --config hf://{repo}/config.yaml --voice speaker.wav \\
  --text "$(python -c 'from africa_g2p import g2p; print(g2p("Akwaaba! Yɛfrɛ me Kofi.", "twi", output="ipa"))')"
```

Tips:
- **Spell out numbers.** Every training clip with a digit in its transcript was removed, so the
  model has never read numerals.
- The voice prompt sets voice, accent and recording conditions; a clean prompt in the target
  language works best.
- The config turns off Pocket TTS's first-letter capitalisation and character replacement, since
  the input is IPA.

## Model

{arch}

## Training data

{hours["total"]:.0f} hours of real speech (no synthetic or voice-converted audio):

| Source | Hours | Notes |
|---|---:|---|
| JW recordings via [AfriSpeech Selector](https://github.com/AfriSpeech/afrispeech-selector) | {src_tot.get("afrispeech", 0):.0f} | read speech, `afrispeech` (JW) source only; native sample rate |
| [AfriSpeech/africa-female-speech-v2](https://huggingface.co/datasets/AfriSpeech/africa-female-speech-v2) | {src_tot.get("africa-female-speech-v2", 0):.0f} | 24 kHz, Demucs-cleaned |
| Ghanaian English | {src_tot.get("ghana-english-tts-clean2", 0):.0f} | |
| [HiFiTTS-2](https://huggingface.co/datasets/nvidia/hifitts-2) (English) | {src_tot.get("hifitts2", 0):.0f} | 200 h slice, training only |

Processing:
- Clips of 1-30 s; **any clip whose transcript contains a number was dropped** (digits and other
  numerals such as Ge'ez).
- Word timings from [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
  (`MahmoudAshraf/mms-300m-1130-forced-aligner`, uroman romanisation) for every language; 99.95% of
  clips aligned. Voice prompts are the start of the same clip, cut at a word boundary.
- Words converted to IPA with africa-g2p's C library (the same tables as the Python package);
  English with espeak-ng (en-us) plus africa-g2p's normalisation. Only languages with an africa-g2p
  IPA table were kept.

## Languages

✓ = UDHR sample on the [samples page]({SPACE}).

| Language | Code | Hours | Sample |
|---|---|---:|:-:|
{rows}

## Limitations

- **Mostly read religious speech.** Most African-language hours are JW readings, so news,
  conversation and technical vocabulary are less natural; low-hour languages (under ~2 h) are weak.
- **Tone:** some africa-g2p tables do not mark tone (e.g. Fon, Ewe, Igbo, Ga), so the model has to
  guess it from context.
- **Loanwords** are converted with the host language's spelling rules ("Japanese" in Twi becomes
  `d͡ʒapanese`).
- No numerals (see above). No language tag: the language comes from the IPA and the voice prompt.
- Voice cloning can reproduce anyone's voice. Do not use it to impersonate people or without consent.

## License and attribution

CC-BY-NC-4.0 (non-commercial), in line with the terms of the JW recordings in the training data.
Built on [Kyutai's Pocket TTS](https://huggingface.co/kyutai/pocket-tts) (CC-BY-4.0) and its
training recipe. Text front end: [africa-g2p](https://github.com/AfriSpeech/africa-g2p).
Trained by [AfriSpeech](https://huggingface.co/AfriSpeech) / Ghana NLP.
"""
    open(out, "w", encoding="utf-8").write(card)
    print("wrote", out)


if __name__ == "__main__":
    main()
