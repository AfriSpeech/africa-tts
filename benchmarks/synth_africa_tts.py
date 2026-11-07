"""Synthesise the benchmark test set with Africa TTS, the way users run it: the africa_tts package,
each language's default voice, temperature 0.3.

Systems: turbo (Africa TTS), guided / unguided (the 24-layer africa-tts-base, cfg 2.0 / 1.0).

Usage: python synth_africa_tts.py BENCH_DIR --system turbo [--device cuda] [--shard i/n]
Writes BENCH_DIR/synth/<system>/<lang>/<id>.wav
"""

import argparse
import json
from pathlib import Path

import scipy.io.wavfile
import torch

from africa_tts import AfricaTTS

SYSTEMS = {"turbo": ("africa-tts", 1.0), "guided": ("africa-tts-base", 2.0),
           "unguided": ("africa-tts-base", 1.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path)
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="i/n: this process does rows i, i+n, ...")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    rows = [json.loads(line) for line in open(a.bench / "testset.jsonl")][i::n]
    model, cfg = SYSTEMS[a.system]
    torch.manual_seed(0)
    tts = AfricaTTS.load(model, cfg=cfg)
    tts.model.to(a.device)
    done = 0
    for r in rows:
        out = a.bench / "synth" / a.system / r["lang"] / f"{r['id']}.wav"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        audio = tts.generate(r["text"], r["lang"])
        scipy.io.wavfile.write(out, tts.sample_rate, audio)
        done += 1
        if done % 100 == 0:
            print(f"{a.system}: {done}", flush=True)
    print(f"{a.system}: done ({done} new)")


if __name__ == "__main__":
    main()
