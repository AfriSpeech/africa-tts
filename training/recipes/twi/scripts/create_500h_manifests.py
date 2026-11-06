"""Subsample 500 hours of 100% pure Twi from the extracted manifests."""

import json
import random
from pathlib import Path


def subsample(input_jsonl: Path, output_jsonl: Path, target_hours: float, seed: int = 42):
    rng = random.Random(seed)
    rows = []
    with input_jsonl.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rng.shuffle(rows)
    cum = 0.0
    picked = []
    target_sec = target_hours * 3600.0
    for r in rows:
        dur = float(r.get("duration", 0.0))
        if cum < target_sec:
            picked.append(r)
            cum += dur
        else:
            break
    with output_jsonl.open("w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{input_jsonl.name} -> {output_jsonl.name}: {len(picked):,} rows, {cum / 3600:.2f}h")
    return picked


def main():
    root = Path("data/twi/manifests")
    out_dir = Path("data/twi/manifests_500h")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 500h 100% Pure Twi: 475h train, 25h valid (5% validation)
    train_twi = subsample(root / "train_twi.jsonl", out_dir / "train_twi.jsonl", 475.0, seed=42)
    valid_twi = subsample(root / "valid_twi.jsonl", out_dir / "valid_twi.jsonl", 25.0, seed=42)

    # Output train.jsonl and valid.jsonl
    for split, data in [("train", train_twi), ("valid", valid_twi)]:
        out_file = out_dir / f"{split}.jsonl"
        with out_file.open("w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dur = sum(float(r["duration"]) for r in data)
        print(f"Pure Twi {out_file.name}: {len(data):,} rows, {dur / 3600:.2f}h")

    # Combined all_500h.jsonl for audio file collection
    all_500 = train_twi + valid_twi
    out_all = out_dir / "all_500h.jsonl"
    with out_all.open("w") as f:
        for r in all_500:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"Total 500h Pure Twi dataset: {len(all_500):,} rows, {sum(float(r['duration']) / 3600 for r in all_500):.2f}h"
    )


if __name__ == "__main__":
    main()
