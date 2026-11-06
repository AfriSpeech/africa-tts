"""Pocket TTS manifests from parquet speech shards (africa-female-speech-v2, Ghanaian English).

Each parquet shard -> one FLAC (clips joined with a short gap) + manifest rows. Keeps the
dataset's own train/validation split (held out by whole source video). Drops clips whose text
has a number (any digit or other numeral) and clips outside 1-30 s.

Usage: python prepare_female_v2.py SRC_DIR OUT_DIR [--workers 16]
Output: OUT_DIR/audio/<config>/<shard>.flac, OUT_DIR/manifests/<config>.jsonl, summary.json
"""

import argparse
import io
import json
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

GAP = 0.08


def has_number(text: str) -> bool:
    return any(unicodedata.category(c) in ("Nd", "No", "Nl") for c in text)


def do_shard(job: tuple[str, str, str, str, str]) -> tuple[list[dict], dict]:
    shard, config, split, out, source = job
    iso = config.rsplit("_", 1)[1]
    flac = Path(out) / "audio" / config / f"{split}-{Path(shard).stem}.flac"
    flac.parent.mkdir(parents=True, exist_ok=True)
    names = pq.ParquetFile(shard).schema_arrow.names
    t = pq.read_table(shard, columns=["id", "text", "audio"] + (["utmos"] if "utmos" in names else []))
    if "utmos" not in names:
        import pyarrow as pa
        t = t.append_column("utmos", pa.array([None] * t.num_rows, type=pa.float64()))
    rows, st = [], {"clips": 0, "number": 0, "length": 0, "bad": 0, "empty": 0}
    writer, sr, pos = None, None, 0.0
    for cid, text, audio, utmos in zip(*(t.column(c).to_pylist() for c in ("id", "text", "audio", "utmos"))):
        text = " ".join((text or "").split())
        if not text:
            st["empty"] += 1
            continue
        if has_number(text):
            st["number"] += 1
            continue
        try:
            wav, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
        except Exception:  # noqa: BLE001
            st["bad"] += 1
            continue
        if writer is None:
            sr = rate
            writer = sf.SoundFile(flac, "w", samplerate=sr, channels=1, format="FLAC", subtype="PCM_16")
            writer.write(np.zeros(int(GAP * sr), dtype=np.float32))
            pos = GAP
        if rate != sr:
            st["bad"] += 1
            continue
        dur = len(wav) / sr
        if not 1.0 <= dur <= 30.0:
            st["length"] += 1
            continue
        writer.write(np.clip(wav, -1, 1))
        writer.write(np.zeros(int(GAP * sr), dtype=np.float32))
        rows.append({"path": str(flac), "start": round(pos, 4), "duration": round(dur, 4),
                     "transcript": text, "language": config, "iso": iso, "subset": config,
                     "source": source, "split": split, "id": cid,
                     "utmos": utmos})
        pos += dur + GAP
        st["clips"] += 1
    if writer is not None:
        writer.close()
    return rows, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--source", default="africa-female-speech-v2")
    ap.add_argument("--valid-every", type=int, default=0,
                    help="no validation shards in the source: hold out every Nth shard instead")
    args = ap.parse_args()
    jobs = []
    for cfg_dir in sorted(p for p in args.src.iterdir() if p.is_dir() and "_" in p.name):
        for k, shard in enumerate(sorted(cfg_dir.glob("*.parquet"))):
            split = "val" if shard.name.startswith("validation") else "train"
            if args.valid_every and k % args.valid_every == args.valid_every - 1:
                split = "val"
            jobs.append((str(shard), cfg_dir.name, split, str(args.out), args.source))
    if not jobs:
        sys.exit(f"no parquet shards under {args.src}")
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    per: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for (rows, st), job in zip(ex.map(do_shard, jobs), jobs):
            cfg = job[1]
            per.setdefault(cfg, []).extend(rows)
            agg = summary.setdefault(cfg, {"clips": 0, "number": 0, "length": 0, "bad": 0, "empty": 0})
            for k, v in st.items():
                agg[k] += v
    for cfg, rows in per.items():
        with open(args.out / "manifests" / f"{cfg}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[cfg]["hours"] = round(sum(r["duration"] for r in rows) / 3600, 2)
        s = summary[cfg]
        print(f"{cfg:20s} {s['clips']:>6} clips {s['hours']:6.1f} h | dropped: number {s['number']}, "
              f"length {s['length']}, bad {s['bad']}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("total:", round(sum(s["hours"] for s in summary.values()), 1), "h")


if __name__ == "__main__":
    main()
