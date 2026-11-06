"""Transcribe benchmark audio (real recordings and every system's synthesis) with Omnilingual ASR.

Language-conditioned (LLM) model; languages it has no code for are transcribed without one
(flagged in the results). Audio longer than the ASR's 40 s limit is cut at 39.9 s and flagged.

Usage: python transcribe.py BENCH_DIR [--model omniASR_LLM_7B_v2] [--systems real,turbo,...]
Writes BENCH_DIR/asr/<system>.jsonl: {"lang", "id", "hyp", "seconds", "cut"}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

ALIAS = {"twi": "aka", "fat": "fat", "mlg": "plt", "daf": "dnj"}  # our code -> ASR iso


def asr_lang(code: str, script: str | None) -> str | None:
    iso = ALIAS.get(code, code)
    cands = [s for s in supported_langs if s.split("_")[0] == iso]
    if script and f"{iso}_{script}" in cands:
        return f"{iso}_{script}"
    return cands[0] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path)
    ap.add_argument("--model", default="omniASR_LLM_7B_v2")
    ap.add_argument("--systems", default=None, help="comma-separated (default: real + all in synth/)")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.bench / "testset.jsonl")]
    systems = (a.systems.split(",") if a.systems else
               ["real"] + sorted(p.name for p in (a.bench / "synth").iterdir() if p.is_dir()))
    pipe = ASRInferencePipeline(model_card=a.model)
    (a.bench / "asr").mkdir(exist_ok=True)
    for system in systems:
        out = a.bench / "asr" / f"{system}.jsonl"
        if out.exists():
            print(f"{system}: exists, skipping")
            continue
        items = []
        for r in rows:
            if system == "real" and not r["real"]:
                continue
            p = (a.bench / r["real"]) if system == "real" else (
                a.bench / "synth" / system / r["lang"] / f"{r['id']}.wav")
            if not p.exists():
                continue
            wav, sr = sf.read(p, dtype="float32", always_2d=True)
            wav = wav.mean(1)
            secs = len(wav) / sr
            cut = secs > 39.9
            items.append((r, {"waveform": wav[: int(39.9 * sr)].astype(np.float32), "sample_rate": sr},
                          secs, cut))
        # sort by length so batches are uniform
        items.sort(key=lambda x: x[2])
        hyps = pipe.transcribe([x[1] for x in items],
                               lang=[asr_lang(x[0]["lang"], x[0].get("script")) for x in items],
                               batch_size=a.batch_size)
        with open(out, "w") as f:
            for (r, _, secs, cut), h in zip(items, hyps):
                f.write(json.dumps({"lang": r["lang"], "id": r["id"], "hyp": h, "seconds": round(secs, 2),
                                    "cut": cut, "asr_lang": asr_lang(r["lang"], r.get("script"))},
                                   ensure_ascii=False) + "\n")
        print(f"{system}: {len(items)} transcribed", flush=True)


if __name__ == "__main__":
    main()
