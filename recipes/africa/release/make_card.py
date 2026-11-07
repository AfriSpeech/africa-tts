"""Write the HF model card (README.md) for africa-tts or africa-tts-turbo.

Usage: python make_card.py {teacher|turbo} LANG_HOURS.json UDHR_META.json OUT/README.md [--bench results.json]
  results.json: from benchmarks/score.py (adds the benchmark section).
"""

import argparse
import json
import math

SPACE = "https://huggingface.co/spaces/AfriSpeech/africa-tts-samples"
DEMO = "https://huggingface.co/spaces/AfriSpeech/africa-tts-demo"
GH = "https://github.com/AfriSpeech/africa-tts"
NICE = {"english_eng": "English", "swahili_swa": "Swahili", "twi_twi": "Twi", "yoruba_yor": "Yoruba",
        "xhosa_xho": "Xhosa"}
SYS = {"real": "Real speech", "turbo": "Africa TTS Turbo (110M)",
         "guided": "Africa TTS, guided (336M)", "unguided": "Africa TTS, unguided (336M)",
         "mms": "MMS-TTS (36M per language)", "omnivoice": "OmniVoice (813M)",
         "orpheus": "Sunbird Orpheus (3.3B)"}

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


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def cells_bold(vals: dict, cols: list[str], blank: str = "—") -> list[str]:
    """CERs as %, the lowest synthetic system in bold (real speech is a reference)."""
    shown = {s: pct(v) for s, v in vals.items() if v is not None and not
             (isinstance(v, float) and math.isnan(v))}
    cands = [float(shown[s]) for s in shown if s != "real"]
    best = min(cands) if len(cands) > 1 else None
    return [(f"**{shown[s]}**" if s != "real" and best is not None and float(shown[s]) == best
             else shown[s]) if s in shown else blank for s in cols]


def bench_section(res: dict, turbo: bool) -> str:
    systems = res["systems"]
    me = "turbo" if turbo else "guided"
    order = [me] + [s for s in ("turbo", "guided", "unguided") if s != me and s in systems]
    order += [s for s in ("mms", "omnivoice", "orpheus") if s in systems]
    cols = [s for s in ["real", *order] if s in systems]
    rows = ["| Languages | " + " | ".join(SYS[s] for s in cols) + " |", "|---|" + "---:|" * len(cols)]
    for comp in res["averages"]:
        if comp["vs"] not in ("mms", "omnivoice", "orpheus", "real"):
            continue
        label = {"mms": "MMS-TTS supports", "omnivoice": "OmniVoice supports",
                 "orpheus": "Sunbird Orpheus supports",
                 "real": "have real recordings"}[comp["vs"]]
        rows.append(f"| {len(comp['languages'])} that {label} | "
                    + " | ".join(cells_bold(comp["cer"], cols, "")) + " |")
    per = ["| Language | " + " | ".join(SYS[s] for s in cols) + " |", "|---|" + "---:|" * len(cols)]
    for c, v in res["languages"].items():
        cells = cells_bold({s: v[s]["cer"] for s in cols if s in v}, cols)
        per.append(f"| {v['name']} `{c}` | " + " | ".join(cells) + " |")
    return f"""## Benchmark

Intelligibility as **character error rate (CER, %, lower is better)**: each system reads the same
held-out sentences (Omnilingual ASR corpus, FLEURS, WAXAL, FLORES+; none in our training data), and
[Omnilingual ASR](https://huggingface.co/facebook/omniASR-LLM-7B) transcribes the audio. *Real
speech* is the ASR on human recordings of the same sentences, a reference for the ASR's own error.
Averages are over the languages each comparison shares; the best synthetic system is in bold.
Africa TTS uses each language's default voice; OmniVoice clones the same clip; MMS-TTS and Sunbird Orpheus use their own voices. Languages
with fewer than 10 test sentences are left out of the averages.

{chr(10).join(rows)}

<details><summary>Per language ({len(res["languages"])} languages)</summary>

{chr(10).join(per)}

</details>

Method, caveats and scripts: [benchmarks/]({GH}/tree/main/benchmarks).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["teacher", "turbo"])
    ap.add_argument("hours")
    ap.add_argument("meta")
    ap.add_argument("out")
    ap.add_argument("--bench")
    a = ap.parse_args()
    hours = json.load(open(a.hours))
    meta = json.load(open(a.meta))
    udhr = {it["g2p"] for it in meta["items"]}
    turbo = a.kind == "turbo"
    repo = "AfriSpeech/africa-tts-turbo" if turbo else "AfriSpeech/africa-tts"
    langs = hours["langs"]
    african = [x for x in langs if x[0] != "eng"]
    rows = "\n".join(
        f"| {nice(n)} | `{g}`{f' (ISO `{RETIRED[g]}`)' if g in RETIRED else ''} | {h:.1f} | "
        f"{'✓' if g in udhr else ''} |" for g, n, h, _ in
        sorted(langs, key=lambda x: (x[0] != "eng", -x[2])))
    short = "africa-tts-turbo" if turbo else "africa-tts"
    if turbo:
        what = (f"Fast text-to-speech for **{len(african)} African languages and English**, with voice "
                f"cloning: the 6-layer version of [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts), "
                f"distilled with guidance built in, faster than real time on a CPU.")
        model = [
            "6-layer [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) FlowLM distilled from the "
            "24-layer Africa TTS (guided, cfg 2.0) with Kyutai's `depth_distill` recipe, 200k steps.",
            "Reads IPA from [africa-g2p](https://github.com/AfriSpeech/africa-g2p) (English: espeak-ng); "
            "the package does the conversion.",
            f"Ready [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) bundle in "
            f"[sherpa-onnx/](https://huggingface.co/{repo}/tree/main/sherpa-onnx) for servers and devices.",
        ]
    else:
        what = (f"Text-to-speech for **{len(african)} African languages and English**, with voice cloning: "
                f"the 24-layer model. [Africa TTS Turbo](https://huggingface.co/AfriSpeech/africa-tts-turbo) "
                f"is its fast 6-layer version.")
        model = [
            "24-layer [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) FlowLM (316M parameters), "
            "fine-tuned from Kyutai's English model with a new 4,000-token IPA tokenizer; step 100k, "
            "chosen by listening.",
            "Reads IPA from [africa-g2p](https://github.com/AfriSpeech/africa-g2p) (English: espeak-ng); "
            "the package does the conversion.",
            "Two modes: **guided** (cfg 2.0, the package default for this model, Python) and "
            f"**unguided** (`cfg=1.0`, also in the [sherpa-onnx bundle](https://huggingface.co/{repo}/tree/main/sherpa-onnx)). "
            f"[How the modes work]({GH}/blob/main/docs/MODES.md).",
        ]
    model.append(f"Trained on {hours['total']:.0f} h of real speech: JW recordings, "
                 "[africa-female-speech-v2](https://huggingface.co/datasets/AfriSpeech/africa-female-speech-v2), "
                 "Ghanaian English and HiFiTTS-2. "
                 f"[Data and recipe]({GH}/blob/main/docs/MODEL.md).")
    bench = bench_section(json.load(open(a.bench)), turbo) if a.bench else ""
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

🎙️ [**Try it live**]({DEMO}) · 🔊 [Samples in {len(udhr)} languages]({SPACE}) · 💻 [Code, server, mobile]({GH})

## Quick start

```bash
pip install "africa-tts @ git+{GH}"   # English also needs espeak-ng
```

```python
from africa_tts import AfricaTTS

tts = AfricaTTS.load("{short}")
audio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")              # default Twi voice
audio = tts.generate("Hello Ghana!", lang="eng", voice="my_voice.wav")  # clone a 3-10 s clip
tts.save(audio, "out.wav")
```

Every language has a default voice. Spell out numbers. Plain pocket-tts works too, with IPA input:
[see how]({GH}/blob/main/docs/MODEL.md#using-plain-pocket-tts).

{bench}
## Model

{chr(10).join(f"- {m}" for m in model)}

<details><summary>Languages ({len(langs)}, with training hours)</summary>

✓ = sample on the [samples page]({SPACE}).

| Language | Code | Hours | Sample |
|---|---|---:|:-:|
{rows}

</details>

## Limitations

Mostly read religious speech, so other registers are less natural; languages with under ~2 h of
data are weak; some G2P tables do not mark tone. Do not use voice cloning to impersonate anyone.
[More]({GH}/blob/main/docs/MODEL.md#limitations).

## License

CC-BY-NC-4.0 (non-commercial), following the JW recordings in the training data. Built on
[Kyutai's Pocket TTS](https://huggingface.co/kyutai/pocket-tts). By [AfriSpeech](https://huggingface.co/AfriSpeech) / Ghana NLP.
"""
    open(a.out, "w", encoding="utf-8").write(card)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
