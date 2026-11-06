"""Pull every AfriSpeech Selector subset into Pocket TTS training format.

Uses the selector's own streaming iterator (clip-length window, digit filter), adds a filter
for non-digit numerals (Ge'ez ፩-፼ etc., Unicode category No/Nl), and takes the train and
validation splits only: each corpus's test split stays out of training so the model can be
evaluated on it.

Per subset: audio/<subset>/chunk_NNN.flac (~1 h each, the subset's native sample rate; clips
joined with a short gap) and manifests/<subset>.jsonl rows {path, start, duration, transcript,
language, iso, country, subset, source, split}. Resumable: subsets with a done/<subset>.json
are skipped.

Usage: python pull_afrispeech.py OUT [--subsets a b ...] [--workers 12] [--min-sec 1 --max-sec 30]
       [--only-numbers --target-sr 24000]   (the number clips, for recovery by alignment)
"""

import argparse
import io
import json
import time
import traceback
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from afrispeech_selector.builder import _iter_rows
from afrispeech_selector.catalog import load_catalog

GAP = 0.08
CHUNK_SEC = 3600.0


def iter_number_rows(subset: str, split: str, min_sec: float, max_sec: float):
    """Like the selector's _iter_rows, but yields ONLY clips whose text has a number (the
    clips _iter_rows drops), for recovering their number-free parts by alignment."""
    from datasets import Audio, load_dataset

    from afrispeech_selector.builder import (
        _audio_len_undecoded, _pick_duration_column, _pick_text_column)
    from afrispeech_selector.catalog import by_subset

    entry = by_subset(subset)
    ds = load_dataset(entry.dataset_id, subset, split={"val": "validation"}.get(split, split),
                      streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    cols = list(ds.features) if ds.features else []
    tc, dc = _pick_text_column(cols), _pick_duration_column(cols)
    for ex in ds:
        text = ex.get(tc, "") if tc else ""
        if not text or not has_number(text):
            continue
        length = float(ex[dc]) if dc and ex.get(dc) is not None else _audio_len_undecoded(ex["audio"])
        if not min_sec <= length <= max_sec:
            continue
        yield {"audio": ex["audio"], "text": text, "language": entry.language,
               "country": entry.country, "length": length, "iso": entry.iso, "subset": subset}


def has_number(text: str) -> bool:
    return any(unicodedata.category(ch) in ("Nd", "No", "Nl") for ch in text)


def has_numeral(text: str) -> bool:
    """Digits are already filtered by the selector (\\d); this catches other numerals."""
    return any(unicodedata.category(ch) in ("No", "Nl") for ch in text)


def pull_subset(subset: str, out: str, min_sec: float, max_sec: float,
                only_numbers: bool = False, target_sr: int | None = None) -> dict:
    out_dir = Path(out)
    done = out_dir / "done" / f"{subset}.json"
    if done.exists():
        return json.loads(done.read_text())
    audio_dir = out_dir / "audio" / subset
    audio_dir.mkdir(parents=True, exist_ok=True)
    stats = {"subset": subset, "clips": 0, "seconds": 0.0, "numeral": 0, "bad_audio": 0,
             "empty_text": 0, "by_split": {}}
    rows: list[dict] = []
    writer = None
    chunk, sr_file, pos = -1, None, 0.0

    def new_chunk(sr: int) -> None:
        nonlocal writer, chunk, sr_file, pos
        if writer is not None:
            writer.close()
        chunk += 1
        sr_file = sr
        writer = sf.SoundFile(audio_dir / f"chunk_{chunk:03d}.flac", "w", samplerate=sr,
                              channels=1, format="FLAC", subtype="PCM_16")
        writer.write(np.zeros(int(GAP * sr), dtype=np.float32))
        pos = GAP

    entry = next(e for e in load_catalog() if e.subset == subset)
    for split in ("train", "val"):
        if entry.split_size(split) <= 0:
            continue
        n_split = 0
        rows_iter = (iter_number_rows(subset, split, min_sec, max_sec) if only_numbers else
                     _iter_rows([subset], split, None, None, min_sec, max_sec, 42, None, None))
        for ex in rows_iter:
            text = " ".join((ex["text"] or "").split())
            if not text:
                stats["empty_text"] += 1
                continue
            if not only_numbers and has_numeral(text):
                stats["numeral"] += 1
                continue
            audio = ex["audio"] or {}
            try:
                data = audio.get("bytes")
                if not data:
                    raise ValueError("no audio bytes")
                wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                wav = wav.mean(axis=1)
            except Exception:  # noqa: BLE001
                stats["bad_audio"] += 1
                continue
            if target_sr and sr != target_sr:
                from scipy.signal import resample_poly
                g = int(np.gcd(sr, target_sr))
                wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
                sr = target_sr
            if writer is None or pos >= CHUNK_SEC:
                new_chunk(sr)
            if sr != sr_file:
                from scipy.signal import resample_poly
                g = int(np.gcd(sr, sr_file))
                wav = resample_poly(wav, sr_file // g, sr // g).astype(np.float32)
            dur = len(wav) / sr_file
            if not (min_sec <= dur <= max_sec):
                continue
            writer.write(np.clip(wav, -1, 1))
            writer.write(np.zeros(int(GAP * sr_file), dtype=np.float32))
            rows.append({"path": str(audio_dir / f"chunk_{chunk:03d}.flac"), "start": round(pos, 4),
                         "duration": round(dur, 4), "transcript": text, "language": ex["language"],
                         "iso": ex["iso"], "country": ex["country"], "subset": subset,
                         "source": entry.dataset, "split": split})
            pos += dur + GAP
            stats["clips"] += 1
            stats["seconds"] += dur
            n_split += 1
        stats["by_split"][split] = n_split
    if writer is not None:
        writer.close()
    (out_dir / "manifests").mkdir(exist_ok=True)
    with open(out_dir / "manifests" / f"{subset}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["hours"] = round(stats.pop("seconds") / 3600, 3)
    done.parent.mkdir(exist_ok=True)
    done.write_text(json.dumps(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--subsets", nargs="*")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--only-numbers", action="store_true",
                    help="pull only the clips whose text has a number (for recovery by alignment)")
    ap.add_argument("--target-sr", type=int, default=None, help="resample on write (e.g. 24000)")
    args = ap.parse_args()
    cat = load_catalog()
    # Largest first so the long ones start early.
    subsets = args.subsets or [e.subset for e in sorted(cat, key=lambda e: -e.hours)]
    t0 = time.time()
    total_h = 0.0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(pull_subset, s, args.out, args.min_sec, args.max_sec, args.only_numbers,
                          args.target_sr): s for s in subsets}
        for i, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                st = fut.result()
                total_h += st["hours"]
                print(f"[{i}/{len(subsets)}] {s}: {st['clips']} clips, {st['hours']:.1f} h "
                      f"(numeral {st['numeral']}, bad audio {st['bad_audio']}) | total {total_h:.0f} h "
                      f"| {(time.time() - t0) / 60:.0f} min", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(subsets)}] {s}: FAILED {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()
