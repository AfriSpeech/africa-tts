"""Write the HF model card (README.md) for AfriSpeech/africa-tts or AfriSpeech/africa-tts-base.

Usage: python make_card.py {main|base} LANG_HOURS.json UDHR_META.json OUT/README.md [--bench results.json]
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
SYS = {"real": "Real speech", "turbo": "Africa TTS (110M)",
         "guided": "Africa TTS Base, guided (336M)", "unguided": "Africa TTS Base, unguided (336M)",
         "mms": "MMS-TTS (36M per language)", "omnivoice": "OmniVoice (813M)"}

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
    cands = [float(shown[s]) for s in cols if s in shown and s != "real"]
    best = min(cands) if len(cands) > 1 else None
    return [(f"**{shown[s]}**" if s != "real" and best is not None and float(shown[s]) == best
             else shown[s]) if s in shown else blank for s in cols]


BENCH_INTRO = """Intelligibility as **character error rate (CER, %, lower is better)**: each system reads the same
held-out sentences (Omnilingual ASR corpus, FLEURS, WAXAL, FLORES+; none in our training data), and
[Omnilingual ASR](https://huggingface.co/facebook/omniASR-LLM-7B) transcribes the audio. *Real
speech* is the ASR on human recordings of the same sentences, a reference for the ASR's own error.
Best synthetic system in bold."""


def bench_section(res: dict, base: bool) -> str:
    systems, L = res["systems"], res["languages"]
    big = {c: v for c, v in L.items() if max(v[s]["n"] for s in v if s != "real" and isinstance(v[s], dict)
                                              and "n" in v[s]) >= 10}
    if base:  # the distillation: Africa TTS against the base model it was distilled from
        cols = [s for s in ("real", "turbo", "guided", "unguided") if s in systems]
        rows = ["| Languages | " + " | ".join(SYS[s] for s in cols) + " |", "|---|" + "---:|" * len(cols)]
        sh = [c for c, v in big.items() if all(s in v for s in ("turbo", "guided", "unguided"))]
        mean = {s: sum(big[c][s]["cer"] for c in sh) / len(sh) for s in ("turbo", "guided", "unguided")}
        rows.append(f"| all {len(sh)} | | " + " | ".join(cells_bold(mean, cols[1:], "")) + " |")
        for comp in res["averages"]:
            if comp["vs"] == "real":
                rows.append(f"| {len(comp['languages'])} with real recordings | "
                            + " | ".join(cells_bold(comp["cer"], cols, "")) + " |")
        return f"""## Benchmark

{BENCH_INTRO} Languages with fewer than 10 test sentences are left out.

{chr(10).join(rows)}

Africa TTS matches the guided base model at a quarter of the layers and one pass per frame.
Comparison with MMS-TTS and OmniVoice: [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts#benchmark).
"""
    cols = [s for s in ("real", "turbo", "mms") if s in systems]
    per = ["| Language | " + " | ".join(SYS[s] for s in cols) + " |", "|---|" + "---:|" * len(cols)]
    for c, v in L.items():
        if "mms" not in v:  # the card compares with MMS-TTS only
            continue
        cells = cells_bold({s: v[s]["cer"] for s in cols if s in v}, cols)
        per.append(f"| {v['name']} `{c}` | " + " | ".join(cells) + " |")
    return f"""## Benchmark

Each system reads the same held-out sentences (Omnilingual ASR corpus, FLEURS, WAXAL, FLORES+;
none in our training data), and [Omnilingual ASR](https://huggingface.co/facebook/omniASR-LLM-7B)
transcribes the audio: a lower **character error rate (CER)** means more intelligible speech.
On the 27 languages both support (at least 10 test sentences each), Africa TTS has the lower
error rate than MMS-TTS in 21:

![CER per language, Africa TTS vs MMS-TTS: lower is better](assets/benchmark_cer.png)

Africa TTS uses each language's default voice; MMS-TTS uses its own voices.

<details><summary>CER per language (%, lower is better; {len(per) - 2} languages)</summary>

*Real speech* is the ASR on human recordings of the same sentences, a reference for the ASR's own
error. Best synthetic system in bold.

{chr(10).join(per)}

</details>

Method, caveats, OmniVoice and all languages: [docs/BENCHMARKS.md]({GH}/blob/main/docs/BENCHMARKS.md).
"""


MAIN_QUICK = '## Quick start\n\n```bash\npip install africa-tts   # English also needs espeak-ng\n```\n\n```python\nfrom africa_tts import AfricaTTS\n\ntts = AfricaTTS.load()\naudio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")              # default Twi voice\naudio = tts.generate("Hello Ghana!", lang="eng", voice="my_voice.wav")  # clone a 3-10 s clip\ntts.save(audio, "out.wav")\n```\n\nThe default engine is [audio.cpp](https://github.com/0xShug0/audio.cpp) on the CPU (the q8_0 GGUF in\n[gguf/](https://huggingface.co/AfriSpeech/africa-tts/tree/main/gguf), no PyTorch);\n`pip install "africa-tts[torch]"` adds the unquantised PyTorch engine, also on a GPU. The first\nrun asks before downloading the model (about 150 MB), then works offline. Output: 24 kHz mono.\nEvery language has a default voice. Spell out numbers. Plain pocket-tts works too, with IPA\ninput: [see how]({GH}/blob/main/docs/MODEL.md#using-plain-pocket-tts).\n\n'.replace('{GH}', 'https://github.com/AfriSpeech/africa-tts')
BASE_USE = '## Use\n\nNot supported by the `africa-tts` package, which runs Africa TTS. For research, the repository\'s\n[`recipes/africa/guidance.py`]({GH}/blob/main/recipes/africa/guidance.py) runs it with guidance:\n\n```python\nimport sys; sys.path.insert(0, "recipes/africa")   # from a clone of {GH}\nfrom guidance import load_base\n\ntts = load_base(cfg=2.0)   # cfg=1.0: one pass per frame\naudio = tts.generate("Akwaaba! Yɛfrɛ me Kofi.", lang="twi")\n```\n\n'.replace('{GH}', 'https://github.com/AfriSpeech/africa-tts')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["main", "base"])
    ap.add_argument("hours")
    ap.add_argument("meta")
    ap.add_argument("out")
    ap.add_argument("--bench")
    a = ap.parse_args()
    hours = json.load(open(a.hours))
    meta = json.load(open(a.meta))
    udhr = {it["g2p"] for it in meta["items"]}
    base = a.kind == "base"
    repo = "AfriSpeech/africa-tts-base" if base else "AfriSpeech/africa-tts"
    langs = hours["langs"]
    african = [x for x in langs if x[0] != "eng"]
    rows = "\n".join(
        f"| {nice(n)} | `{g}`{f' (ISO `{RETIRED[g]}`)' if g in RETIRED else ''} | {h:.1f} | "
        f"{'✓' if g in udhr else ''} |" for g, n, h, _ in
        sorted(langs, key=lambda x: (x[0] != "eng", -x[2])))
    if not base:
        what = (f"Fast text-to-speech for **{len(african)} African languages and English**, with voice "
                "cloning, faster than real time on a CPU: runs on audio.cpp by default (**144 MB, no "
                "PyTorch, first streamed audio after about 0.1 s**), 110M parameters.")
        model = [
            "6-layer [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) FlowLM (110M parameters with "
            "the Mimi codec), distilled with Kyutai's `depth_distill` recipe from the guided 24-layer "
            "[Africa TTS Base](https://huggingface.co/AfriSpeech/africa-tts-base), so guidance is in its "
            "weights; 200k steps.",
            "Reads IPA from [africa-g2p](https://github.com/AfriSpeech/africa-g2p) (English: espeak-ng); "
            "the package does the conversion.",
            "Formats: `model.safetensors` (PyTorch, fp32), `gguf/africa-tts-q8_0.gguf` (audio.cpp, "
            "the default CPU engine) and an int8 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) "
            f"bundle for phones ([sherpa-onnx/](https://huggingface.co/{repo}/tree/main/sherpa-onnx)).",
        ]
    else:
        what = (f"The 24-layer base model that [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts) "
                f"was distilled from ({len(african)} African languages and English), released for "
                "research. **To synthesise speech, use [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts)**: "
                "it is as intelligible and several times faster.")
        model = [
            "24-layer [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) FlowLM (336M parameters with "
            "the Mimi codec), fine-tuned from Kyutai's English model with a new 4,000-token IPA "
            "tokenizer; step 100k, chosen by listening.",
            "Reads IPA from [africa-g2p](https://github.com/AfriSpeech/africa-g2p) (English: espeak-ng).",
            "Best with classifier-free guidance 2.0 (two passes per frame); an unguided "
            f"[sherpa-onnx bundle](https://huggingface.co/{repo}/tree/main/sherpa-onnx) is included.",
        ]
    model.append(f"Trained on {hours['total']:.0f} h of real speech: AfriSpeech Selector recordings, "
                 "[africa-female-speech-v2](https://huggingface.co/datasets/AfriSpeech/africa-female-speech-v2), "
                 "Ghanaian English and HiFiTTS-2. "
                 f"[Data and recipe]({GH}/blob/main/docs/MODEL.md).")
    bench = bench_section(json.load(open(a.bench)), base) if a.bench else ""
    quick = BASE_USE if base else MAIN_QUICK
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

# {"Africa TTS Base" if base else "Africa TTS"}

{what}

🎙️ [**Try it live**]({DEMO}) · 🔊 [Samples in {len(udhr)} languages]({SPACE}) · 💻 [Code, server, mobile]({GH})

{quick}{bench}
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

CC-BY-NC-4.0 (non-commercial), following the licences of the training recordings. Built on
[Kyutai's Pocket TTS](https://huggingface.co/kyutai/pocket-tts). By [AfriSpeech](https://huggingface.co/AfriSpeech) / Ghana NLP.
"""
    open(a.out, "w", encoding="utf-8").write(card)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
