"""Convert dataset audio to a target speaker using AfriSpeech/africa-female-vc.

Runs N parallel conversion workers on GPU. Each worker converts complete episode
FLAC files and writes 24 kHz mono FLACs, preserving chunk timings for alignment.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
from scipy.signal import resample_poly

TARGET_SR = 24_000


def sf_save(filepath, src, sample_rate, **kwargs):
    audio_np = src.detach().cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T
    sf.write(filepath, audio_np, sample_rate)


torchaudio.save = sf_save


def resample_to_24k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return audio
    gcd = int(np.gcd(orig_sr, TARGET_SR))
    up = TARGET_SR // gcd
    down = orig_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def worker_loop(
    file_list: list[Path],
    out_dir: Path,
    voice: str,
    steps: int,
    worker_id: int,
    progress_q: mp.Queue,
):
    torchaudio.save = sf_save
    from africa_female_vc.engine import VoiceConverter

    vc = VoiceConverter(voice=voice, diffusion_steps=steps, install_deps=False)
    vc.load()

    for path in file_list:
        dst = out_dir / path.name
        if dst.exists() and dst.stat().st_size > 1024:
            progress_q.put((worker_id, path.name, 0.0, True))
            continue

        try:
            converted, sr = vc.convert_file(str(path.resolve()), diffusion_steps=steps)
            converted = resample_to_24k(converted, sr)
            # Clip to valid range
            converted = np.clip(converted, -1.0, 1.0)
            sf.write(dst, converted, TARGET_SR, format="FLAC", subtype="PCM_16")
            dur = len(converted) / TARGET_SR
            progress_q.put((worker_id, path.name, dur, False))
        except Exception as exc:
            print(
                f"[Worker {worker_id}] Error converting {path.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            progress_q.put((worker_id, path.name, -1.0, False))

    progress_q.put((worker_id, None, 0.0, False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir", default="data/twi/audio", help="directory of source episode FLAC files"
    )
    ap.add_argument(
        "--output-dir", default="data/twi/audio_vc", help="directory to write converted FLACs"
    )
    ap.add_argument(
        "--manifest", default=None, help="optional jsonl manifest to filter episode files"
    )
    ap.add_argument("--voice", default="clear-high-slow")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        episodes = set()
        with open(args.manifest) as f:
            for line in f:
                if line.strip():
                    episodes.add(Path(json.loads(line)["path"]).name)
        files = sorted([input_dir / ep for ep in episodes if (input_dir / ep).exists()])
    else:
        files = sorted(input_dir.glob("*.flac"))

    if not files:
        print(f"No audio files found to convert in {input_dir}")
        return

    # Check already done
    remaining = [
        f
        for f in files
        if not (out_dir / f.name).exists() or (out_dir / f.name).stat().st_size <= 1024
    ]
    print(
        f"Total files: {len(files):,}, Already converted: {len(files) - len(remaining):,}, To convert: {len(remaining):,}"
    )

    if not remaining:
        print("All files already converted!")
        return

    num_workers = min(args.workers, len(remaining))
    splits = [remaining[i::num_workers] for i in range(num_workers)]
    q = mp.Queue()

    procs = []
    for i in range(num_workers):
        p = mp.Process(target=worker_loop, args=(splits[i], out_dir, args.voice, args.steps, i, q))
        p.start()
        procs.append(p)

    active_workers = num_workers
    done_count = len(files) - len(remaining)
    total_audio_sec = 0.0
    start_time = time.time()

    while active_workers > 0:
        wid, name, dur, was_cached = q.get()
        if name is None:
            active_workers -= 1
            continue
        if was_cached:
            continue
        if dur > 0:
            done_count += 1
            total_audio_sec += dur
            wall = time.time() - start_time
            speed = total_audio_sec / wall if wall > 0 else 0
            pct = done_count / len(files) * 100
            rem_sec = (
                (len(files) - done_count)
                * (total_audio_sec / max(1, (done_count - (len(files) - len(remaining)))))
                / max(1e-3, speed)
            )
            print(
                f"[{done_count:,}/{len(files):,} ({pct:.1f}%)] Converted {name} ({dur:.1f}s) | "
                f"Speed: {speed:.1f}x real-time | Total conv: {total_audio_sec / 3600:.2f}h | "
                f"ETA: {rem_sec / 3600:.1f}h",
                flush=True,
            )

    for p in procs:
        p.join()

    print(f"Done! Converted {len(files):,} files to {out_dir}")


if __name__ == "__main__":
    main()
