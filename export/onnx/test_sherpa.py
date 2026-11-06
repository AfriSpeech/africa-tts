#!/usr/bin/env python3
"""End-to-end synthesis with the real sherpa-onnx runtime from an exported Africa TTS folder.

    python test_sherpa.py --onnx-dir OUT --voice voice.wav --out-dir test_out [--g2p-src PATH]

Text goes through africa-g2p to IPA first (the model reads IPA), then sherpa-onnx.
Runs three variants: fp32, int8 (all int8) and mixed (int8 LM + decoder, fp32 encoder and
text conditioner, as in sherpa-onnx's own Pocket TTS bundles).
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SENTENCES = [
    ("twi", "Akwaaba! Yɛfrɛ me Kofi, na me firi Kumase. Ɛnnɛ yɛbɛkɔ fie no akyiri akɔhwɛ yɛn nanabaa."),
    ("eng", "Hello Ghana, how are you today? We are testing the new speech model on a phone."),
]
VARIANTS = {
    "fp32": dict(lm_main="lm_main.onnx", lm_flow="lm_flow.onnx", encoder="encoder.onnx",
                 decoder="decoder.onnx", text_conditioner="text_conditioner.onnx"),
    "int8": dict(lm_main="lm_main.int8.onnx", lm_flow="lm_flow.int8.onnx", encoder="encoder.int8.onnx",
                 decoder="decoder.int8.onnx", text_conditioner="text_conditioner.int8.onnx"),
    "mixed": dict(lm_main="lm_main.int8.onnx", lm_flow="lm_flow.int8.onnx", encoder="encoder.onnx",
                  decoder="decoder.int8.onnx", text_conditioner="text_conditioner.onnx"),
}
_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def to_ipa(text: str, lang: str) -> str:
    from africa_g2p import g2p

    if lang != "eng":
        return g2p(text, lang, output="ipa")
    words = []
    for w in text.split():
        lead, core, trail = _PUNCT.match(w).groups()
        words.append(lead + (g2p(core, "eng", output="ipa").replace(" ", "") if core else "") + trail)
    return " ".join(words)


def make_tts(d: Path, files: dict, threads: int):
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                **{k: str(d / v) for k, v in files.items()},
                vocab_json=str(d / "vocab.json"),
                token_scores_json=str(d / "token_scores.json"),
            ),
            num_threads=threads,
            provider="cpu",
        )
    )
    assert cfg.validate()
    return sherpa_onnx.OfflineTts(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--g2p-src", default=None, help="africa-g2p src dir if not pip-installed")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--variants", default="fp32,int8,mixed")
    a = ap.parse_args()
    if a.g2p_src:
        sys.path.insert(0, a.g2p_src)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ref, sr = sf.read(a.voice, dtype="float32", always_2d=True)
    ref = ref[:, 0]
    ipa = [(lang, to_ipa(text, lang)) for lang, text in SENTENCES]
    for lang, s in ipa:
        print(f"{lang}: {s}")
    import sherpa_onnx

    for name in a.variants.split(","):
        t0 = time.time()
        tts = make_tts(a.onnx_dir, VARIANTS[name], a.threads)
        load = time.time() - t0
        for lang, text in ipa:
            g = sherpa_onnx.GenerationConfig()
            g.reference_audio = ref
            g.reference_sample_rate = sr
            g.num_steps = 1  # LSD decode steps (pocket-tts default)
            g.extra = {"temperature": "0.3", "seed": "42"}
            t0 = time.time()
            audio = tts.generate(text, g)
            el = time.time() - t0
            x = np.asarray(audio.samples, np.float32)
            dur = len(x) / audio.sample_rate
            rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
            out = a.out_dir / f"sherpa_{name}_{lang}.wav"
            sf.write(out, x, audio.sample_rate)
            print(f"{name:5s} {lang}: {dur:5.2f}s audio, rms {rms:.4f}, peak {np.abs(x).max() if len(x) else 0:.3f}, "
                  f"{el:5.2f}s compute (RTF {el / max(dur, 1e-6):.2f}, load {load:.1f}s) -> {out.name}")


if __name__ == "__main__":
    main()
