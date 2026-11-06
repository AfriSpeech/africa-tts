"""Build Pocket TTS manifests from the Ghana podcasts Twi/English parquet dataset.

Rows are grouped by episode_id so a training sample's voice prompt and target
come from the same speaker. Selected chunks are concatenated (with a short
silence gap) into one 24 kHz mono FLAC per episode; the manifest windows point
back into those files.

Usage:
  python data/prepare_twi.py --twi-hours 700 --en-hours 300
  python data/prepare_twi.py --twi-hours 1 --en-hours 0.5 --limit-files 2 --smoke
"""

import argparse
import io
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

SR = 24_000
GAP_SEC = 0.08
META_COLS = [
    "transcription",
    "language",
    "duration",
    "sample_rate",
    "channels",
    "episode_id",
    "chunk_index",
    "podcast_name",
    "station_name",
    "episode_title",
    "start_time",
    "end_time",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def norm_lang(x: str) -> str:
    x = str(x).strip().lower()
    if x in {"twi", "ak", "akan", "asante", "akuapem"}:
        return "twi"
    if x in {"en", "eng", "english"}:
        return "english"
    return x


def parquet_files(root: Path, limit: int | None) -> list[Path]:
    files = sorted(root.glob("data/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {root}")
    if limit:
        files = files[:limit]
    return files


def scan_metadata(root: Path, files: list[Path], cache: Path) -> pd.DataFrame:
    if cache.exists():
        log(f"using cached metadata {cache}")
        return pd.read_parquet(cache)
    frames = []
    for i, f in enumerate(files, 1):
        t = pq.read_table(f, columns=META_COLS)
        df = t.to_pandas()
        df["parquet_file"] = f.name
        df["row_idx"] = np.arange(len(df), dtype=np.int64)
        frames.append(df)
        if i % 25 == 0 or i == len(files):
            log(f"scanned {i}/{len(files)} files, {sum(len(x) for x in frames):,} rows")
    meta = pd.concat(frames, ignore_index=True)
    meta["language"] = meta["language"].map(norm_lang)
    meta = meta[
        meta["transcription"].notna()
        & (meta["transcription"].str.strip().str.len() > 0)
        & (meta["duration"] >= 2.0)
        & (meta["duration"] <= 30.0)
    ].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(cache, index=False)
    log(f"metadata: {len(meta):,} usable rows, {meta['duration'].sum() / 3600:.1f} h")
    return meta


def select_rows(meta: pd.DataFrame, twi_hours: float, en_hours: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    picked = []
    for lang, quota_h in (("twi", twi_hours), ("english", en_hours)):
        part = meta[meta["language"] == lang].reset_index(drop=True)
        if part.empty:
            log(f"WARNING: no {lang} rows found")
            continue
        order = rng.permutation(len(part))
        part = part.iloc[order].reset_index(drop=True)
        cum = part["duration"].cumsum() / 3600.0
        take = int(np.searchsorted(cum.to_numpy(), quota_h, side="left")) + 1
        take = min(take, len(part))
        chosen = part.iloc[:take].copy()
        chosen["split_lang"] = lang
        picked.append(chosen)
        log(
            f"{lang}: selected {len(chosen):,} rows / {chosen['duration'].sum() / 3600:.1f} h "
            f"(available {part['duration'].sum() / 3600:.1f} h)"
        )
    if not picked:
        raise SystemExit("nothing selected")
    sel = pd.concat(picked, ignore_index=True)
    # Split by episode so no episode leaks between train and valid.
    eps = sel["episode_id"].astype(str).unique()
    rng.shuffle(eps)
    valid_eps = set(eps[: max(1, int(round(0.05 * len(eps))))])
    sel["split"] = np.where(sel["episode_id"].astype(str).isin(valid_eps), "valid", "train")
    for split in ("train", "valid"):
        p = sel[sel["split"] == split]
        by = p.groupby("language")["duration"].sum() / 3600
        log(f"  {split}: {len(p):,} rows, {p['duration'].sum() / 3600:.1f} h, {by.to_dict()}")
    return sel


class HandleCache:
    def __init__(self, max_open: int = 512):
        self.max_open = max_open
        self.handles: OrderedDict[str, int] = OrderedDict()

    def get(self, path: Path) -> int:
        key = str(path)
        fd = self.handles.pop(key, None)
        if fd is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self.handles[key] = fd
        if len(self.handles) > self.max_open:
            _, old = self.handles.popitem(last=False)
            os.close(old)
        return fd

    def close(self) -> None:
        for fd in self.handles.values():
            os.close(fd)
        self.handles.clear()


def extract(sel: pd.DataFrame, files: list[Path], out_dir: Path) -> pd.DataFrame:
    pcm_dir = out_dir / "pcm"
    pcm_dir.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, set[int]] = {}
    sel_lookup: dict[tuple[str, int], dict] = {}
    for row in sel.itertuples(index=False):
        key = (row.parquet_file, int(row.row_idx))
        by_file.setdefault(row.parquet_file, set()).add(int(row.row_idx))
        sel_lookup[key] = row._asdict()
    # chunk records in append order
    records: list[dict] = []
    lengths: dict[str, int] = {}
    handles = HandleCache()
    done_rows = 0
    bad = 0
    for fi, f in enumerate(files, 1):
        wanted = by_file.get(f.name)
        if not wanted:
            continue
        pf = pq.ParquetFile(f)
        row_idx = 0
        for batch in pf.iter_batches(columns=["audio"], batch_size=1024):
            audio_col = batch.column("audio")
            for j in range(len(audio_col)):
                idx = row_idx + j
                if idx not in wanted:
                    continue
                try:
                    item = audio_col[j].as_py()
                    data = item["bytes"]
                    if not data:
                        raise ValueError("empty audio")
                    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                    mono = wav.mean(axis=1)
                    if sr != SR:
                        g = int(np.gcd(sr, SR))
                        mono = resample_poly(mono, SR // g, sr // g).astype(np.float32)
                    pcm = np.clip(mono, -1.0, 1.0)
                    pcm = (pcm * 32767.0).astype("<i2")
                except Exception as exc:  # noqa: BLE001
                    bad += 1
                    if bad <= 10:
                        log(f"WARNING skip {f.name}:{idx}: {exc}")
                    continue
                m = sel_lookup.get((f.name, idx))
                if m is None:
                    continue
                ep = str(m["episode_id"])
                pcm_path = pcm_dir / f"{ep}.pcm"
                fd = handles.get(pcm_path)
                gap = np.zeros(int(round(GAP_SEC * SR)), dtype="<i2").tobytes()
                start_sample = lengths.get(ep)
                if start_sample is None:
                    os.write(fd, gap)
                    start_sample = len(gap) // 2
                os.write(fd, pcm.tobytes())
                n = len(pcm)
                os.write(fd, gap)
                lengths[ep] = start_sample + n + len(gap) // 2
                records.append(
                    {
                        "path": str((out_dir / "audio" / f"{ep}.flac").resolve()),
                        "start": start_sample / SR,
                        "duration": n / SR,
                        "transcript": str(m["transcription"]).strip(),
                        "language": str(m["language"]),
                        "episode_id": ep,
                        "chunk_index": int(m["chunk_index"]),
                        "podcast_name": m["podcast_name"],
                        "station_name": m["station_name"],
                        "episode_title": m["episode_title"],
                        "split": m["split"],
                    }
                )
                done_rows += 1
            row_idx += len(audio_col)
        if fi % 10 == 0 or fi == len(files):
            log(f"extracted {fi}/{len(files)} files, {done_rows:,} chunks, {bad} bad")
    handles.close()
    log(f"converting {len(lengths):,} episode PCM files to FLAC")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for k, (ep, expected) in enumerate(lengths.items(), 1):
        src = pcm_dir / f"{ep}.pcm"
        dst = audio_dir / f"{ep}.flac"
        with sf.SoundFile(
            dst, "w", samplerate=SR, channels=1, format="FLAC", subtype="PCM_16"
        ) as out:
            with open(src, "rb") as fh:
                while block := fh.read(SR * 2 * 60):
                    out.write(np.frombuffer(block, dtype="<i2"))
        if k % 500 == 0 or k == len(lengths):
            log(f"  flac {k}/{len(lengths)}")
        src.unlink()
    try:
        pcm_dir.rmdir()
    except OSError:
        pass
    return pd.DataFrame(records)


def write_manifests(df: pd.DataFrame, out_dir: Path) -> None:
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "start",
        "duration",
        "transcript",
        "language",
        "episode_id",
        "chunk_index",
        "podcast_name",
        "station_name",
        "episode_title",
    ]
    for split in ("train", "valid"):
        for lang in ("twi", "english"):
            part = df[(df["split"] == split) & (df["language"] == lang)]
            p = manifest_dir / f"{split}_{lang}.jsonl"
            with open(p, "w") as f:
                for row in part[fields].to_dict("records"):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"wrote {p} ({len(part):,} rows, {part['duration'].sum() / 3600:.1f} h)")
        part = df[df["split"] == split]
        p = manifest_dir / f"{split}.jsonl"
        with open(p, "w") as f:
            for row in part[fields].to_dict("records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"wrote {p} ({len(part):,} rows, {part['duration'].sum() / 3600:.1f} h)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ghana-podcasts")
    ap.add_argument("--out", default="data/twi")
    ap.add_argument("--twi-hours", type=float, default=700.0)
    ap.add_argument("--en-hours", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument(
        "--smoke", action="store_true", help="small quotas, keep metadata cache per limit"
    )
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = parquet_files(root, args.limit_files)
    cache = out / f"metadata{'_smoke' if args.smoke else ''}.parquet"
    meta = scan_metadata(root, files, cache)
    log(f"languages: {meta.groupby('language')['duration'].sum().div(3600).round(1).to_dict()}")
    if args.scan_only:
        return
    sel = select_rows(meta, args.twi_hours, args.en_hours, args.seed)
    sel.to_parquet(out / f"selected{'_smoke' if args.smoke else ''}.parquet", index=False)
    if args.select_only:
        return
    df = extract(sel, files, out)
    write_manifests(df, out)
    log(f"done: {len(df):,} chunks, {df['duration'].sum() / 3600:.1f} h")


if __name__ == "__main__":
    main()
