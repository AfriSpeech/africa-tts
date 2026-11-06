"""CPU speed of Africa TTS: how much faster than real time each way of running it is.

Measures, on this machine's CPU, for a set of sentences in several languages:
  - real-time factor (RTF = compute seconds / audio seconds; "x real time" = 1 / RTF)
  - time to first audio when streaming (Python package)
for Turbo and the 24-layer model, in the Python package (PyTorch) and in sherpa-onnx (int8).

The Python package runs PyTorch on one thread (pocket-tts's default); --threads sets sherpa-onnx's.

Usage: python speed.py [--threads 4] [--out speed.json]
Needs: africa-tts, sherpa-onnx; downloads the models and sherpa-onnx bundles from Hugging Face.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

SENTENCES = [
    ("twi", "Akwaaba! Yɛfrɛ me Kofi, na mefiri Kumase. Ɛnnɛ yɛbɛkɔ fie no akɔhwɛ yɛn nanabaa."),
    ("hau", "Sannu da zuwa. Yau za mu je kasuwa domin mu sayi abinci da kayan makaranta."),
    ("yor", "Ẹ kú àárọ̀. A ó lọ sí ọjà lónìí láti ra oúnjẹ àti aṣọ tuntun fún àwọn ọmọdé."),
    ("swh", "Habari za asubuhi. Leo tutakwenda sokoni kununua chakula na vitabu vya shule."),
    ("zul", "Sawubona. Namuhla sizoya emakethe ukuyothenga ukudla nezincwadi zesikole."),
    ("amh", "እንደምን አደርክ። ዛሬ ወደ ገበያ ሄደን ምግብና የትምህርት ቤት መጻሕፍት እንገዛለን።"),
    ("ewe", "Ŋdi na mi. Egbe míayi asi me be míaƒle nuɖuɖu kple sukuwo ƒe agbalẽwo."),
    ("eng", "Good morning. Today we are going to the market to buy food and new school books."),
]


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def bench_python(model: str, cfg: float) -> dict:
    from africa_tts import AfricaTTS

    tts = AfricaTTS.load(model, cfg=cfg)  # pocket-tts runs PyTorch on 1 thread (its default)
    from africa_tts.frontend import to_ipa

    tts.generate("Akwaaba.", "twi")  # warm-up
    for lang, _ in SENTENCES:  # voice prompts are encoded once per voice and cached: not timed
        tts._voice_state(tts.default_voice(lang))
    audio_s = compute_s = 0.0
    first = []
    for lang, text in SENTENCES:
        t0 = time.perf_counter()
        audio = tts.generate(text, lang)
        compute_s += time.perf_counter() - t0
        audio_s += len(audio) / tts.sample_rate
        t0 = time.perf_counter()  # time to first audio when streaming
        stream = tts.model.generate_audio_stream(tts._voice_state(tts.default_voice(lang)), to_ipa(text, lang))
        next(iter(stream))
        first.append(time.perf_counter() - t0)
        for _ in stream:  # drain
            pass
    out = {"audio_s": round(audio_s, 2), "compute_s": round(compute_s, 2),
           "rtf": round(compute_s / audio_s, 3), "x_realtime": round(audio_s / compute_s, 2)}
    out["first_audio_s"] = round(sorted(first)[len(first) // 2], 2)  # median
    return out


def bench_sherpa(repo: str, threads: int) -> dict:
    import numpy as np
    import sherpa_onnx
    import soundfile as sf
    from huggingface_hub import snapshot_download

    from africa_tts.frontend import to_ipa

    b = Path(snapshot_download(repo, allow_patterns=["sherpa-onnx/*"])) / "sherpa-onnx"
    pocket = sherpa_onnx.OfflineTtsPocketModelConfig(
        lm_main=str(b / "lm_main.int8.onnx"), lm_flow=str(b / "lm_flow.int8.onnx"),
        encoder=str(b / "encoder.onnx"), decoder=str(b / "decoder.int8.onnx"),
        text_conditioner=str(b / "text_conditioner.onnx"),
        vocab_json=str(b / "vocab.json"), token_scores_json=str(b / "token_scores.json"))
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        pocket=pocket, num_threads=threads, provider="cpu")))

    def gen(text: str, lang: str):
        wav, sr = sf.read(b / "voices" / f"{lang}.wav", dtype="float32")
        g = sherpa_onnx.GenerationConfig()
        g.reference_audio, g.reference_sample_rate = wav, sr
        g.num_steps, g.silence_scale = 1, 1.0
        g.extra = {"temperature": "0.3"}
        return np.asarray(tts.generate(to_ipa(text, lang), g).samples)

    gen("Akwaaba.", "twi")
    audio_s = compute_s = 0.0
    for lang, text in SENTENCES:
        t0 = time.perf_counter()
        a = gen(text, lang)
        compute_s += time.perf_counter() - t0
        audio_s += len(a) / 24000
    return {"audio_s": round(audio_s, 2), "compute_s": round(compute_s, 2),
            "rtf": round(compute_s / audio_s, 3), "x_realtime": round(audio_s / compute_s, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", type=Path, default=Path("speed.json"))
    ap.add_argument("--skip", default="", help="comma-separated runs to skip")
    a = ap.parse_args()
    runs = {
        "turbo_sherpa_int8": lambda: bench_sherpa("AfriSpeech/africa-tts-turbo", a.threads),
        "turbo_python": lambda: bench_python("africa-tts-turbo", 1.0),
        "24l_unguided_sherpa_int8": lambda: bench_sherpa("AfriSpeech/africa-tts", a.threads),
        "24l_unguided_python": lambda: bench_python("africa-tts", 1.0),
        "24l_guided_python": lambda: bench_python("africa-tts", 2.0),
    }
    res = {"cpu": cpu_name(), "threads": a.threads, "sentences": len(SENTENCES), "runs": {}}
    for name, fn in runs.items():
        if name in a.skip.split(","):
            continue
        res["runs"][name] = fn()
        print(name, res["runs"][name], flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
