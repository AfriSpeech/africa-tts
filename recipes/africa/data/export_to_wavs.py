"""Export sliced individual WAV files and LJSpeech-compatible metadata.csv.

Useful for training third-party TTS engines (e.g. Coqui TTS, Piper, VITS,
Matcha-TTS, FastSpeech2) that require a folder of individual audio files.

Usage:
    python export_to_wavs.py --out OUT_DIR [--split train] [--lang twi] [--sample-rate 24000] [--max-clips 1000]
"""

import argparse
import csv
import json
from pathlib import Path
import soundfile as sf
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sliced audio clips to LJSpeech format.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--split", default="train", choices=["train", "valid"], help="Split to export")
    parser.add_argument("--lang", default=None, help="Filter by language code (e.g. twi, yor, swh, eng)")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Target sample rate (Hz)")
    parser.add_argument("--max-clips", type=int, default=None, help="Maximum number of clips to export")
    args = parser.parse_args()

    manifest = args.root / f"{args.split}.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    wavs_dir = args.out / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "metadata.csv"

    print(f"Reading {manifest}...")
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if args.lang is None or r.get("g2p", "").lower() == args.lang.lower() or r.get("iso", "").lower() == args.lang.lower():
                rows.append(r)
                if args.max_clips and len(rows) >= args.max_clips:
                    break

    print(f"Exporting {len(rows):,} clips to {args.out}...")
    exported = 0
    with open(meta_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        writer.writerow(["id", "text", "ipa", "speaker", "duration"])

        for i, row in enumerate(rows):
            lang_tag = args.lang or row.get("g2p", "clip")
            clip_id = f"{lang_tag}_{i:06d}"
            out_wav = wavs_dir / f"{clip_id}.wav"

            src_wav = args.root / row["path"]
            info = sf.info(str(src_wav))
            sr = info.samplerate
            start_frame = int(row["start"] * sr)
            frame_count = int(row["duration"] * sr)

            audio, _ = sf.read(
                str(src_wav),
                start=start_frame,
                frames=frame_count,
                dtype="float32",
                always_2d=True,
            )
            audio = audio.mean(1)

            if args.sample_rate and sr != args.sample_rate:
                from scipy.signal import resample_poly
                g = np.gcd(sr, args.sample_rate)
                audio = resample_poly(audio, args.sample_rate // g, sr // g).astype(np.float32)
                out_sr = args.sample_rate
            else:
                out_sr = sr

            sf.write(str(out_wav), audio, out_sr, subtype="PCM_16")

            text = row.get("transcript_orig", row.get("text", "")).strip()
            ipa = row.get("transcript", "").strip()
            speaker = row.get("subset", "speaker0")
            dur = f"{len(audio)/out_sr:.2f}"

            writer.writerow([clip_id, text, ipa, speaker, dur])
            exported += 1
            if exported % 1000 == 0:
                print(f"  Exported {exported:,}/{len(rows):,} clips...")

    print(f"Done! Exported {exported:,} WAV files in {wavs_dir} and metadata in {meta_path}")


if __name__ == "__main__":
    main()
