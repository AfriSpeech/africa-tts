"""Africa TTS through sherpa-onnx (the on-device runtime), from Python.

    pip install sherpa-onnx soundfile "africa-g2p[english]>=0.3.4"
    huggingface-cli download AfriSpeech/africa-tts --include "sherpa-onnx/*" --local-dir .
    python speak_sherpa.py sherpa-onnx "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o out.wav

The text is converted to IPA with africa-g2p (the same IPA the C front end produces on device),
then synthesised in the language's bundled default voice (or --voice).
"""

import argparse
import re

import numpy as np
import sherpa_onnx
import soundfile as sf
from africa_g2p import g2p

_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def to_ipa(text: str, lang: str) -> str:
    if lang != "eng":
        return g2p(text, lang, output="ipa")
    out = []
    for w in text.split():
        lead, core, trail = _PUNCT.match(w).groups()
        out.append(lead + (g2p(core, "eng", output="ipa").replace(" ", "") if core else "") + trail)
    return " ".join(out)


def load(bundle: str, threads: int = 2) -> sherpa_onnx.OfflineTts:
    pocket = sherpa_onnx.OfflineTtsPocketModelConfig(
        lm_main=f"{bundle}/lm_main.int8.onnx", lm_flow=f"{bundle}/lm_flow.int8.onnx",
        encoder=f"{bundle}/encoder.onnx", decoder=f"{bundle}/decoder.int8.onnx",
        text_conditioner=f"{bundle}/text_conditioner.onnx",
        vocab_json=f"{bundle}/vocab.json", token_scores_json=f"{bundle}/token_scores.json")
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        pocket=pocket, num_threads=threads, provider="cpu"))
    return sherpa_onnx.OfflineTts(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("text")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--voice", help="WAV to clone (default: bundle/voices/<lang>.wav)")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    tts = load(a.bundle, a.threads)
    ref, sr = sf.read(a.voice or f"{a.bundle}/voices/{a.lang}.wav", dtype="float32")
    g = sherpa_onnx.GenerationConfig()
    g.reference_audio = ref if ref.ndim == 1 else ref.mean(1)
    g.reference_sample_rate = sr
    g.num_steps = 1
    g.silence_scale = 1.0
    g.extra = {"temperature": "0.3"}
    ipa = to_ipa(a.text, a.lang)
    audio = tts.generate(ipa, g)
    sf.write(a.out, np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)
    print(f"{ipa}\n-> {a.out} ({len(audio.samples) / audio.sample_rate:.1f} s)")


if __name__ == "__main__":
    main()
