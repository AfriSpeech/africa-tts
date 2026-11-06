"""Merge per-language aligned manifests and drop rows with no usable word cut."""

import argparse
import json
import random
from pathlib import Path


def has_cut(entry: dict) -> bool:
    words = entry.get("words")
    duration = float(entry.get("duration", 0.0))
    if not words or duration < 2.0:
        return False
    for prev, cur in zip(words, words[1:]):
        if prev.get("end") is None or cur.get("start") is None:
            continue
        cut = 0.5 * (float(prev["end"]) + float(cur["start"]))
        if cut >= 1.0 and duration - cut >= 1.0:
            return True
    return False


def load(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if has_cut(e):
                rows.append(e)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path("data")
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)
    for split in ("train", "valid"):
        rows = []
        for lang in ("twi", "english"):
            p = root / f"{split}_{lang}_aligned.jsonl"
            got = load(p)
            print(f"{p}: {len(got):,} usable")
            rows.extend(got)
        rng.shuffle(rows)
        out = out_dir / f"{split}_aligned.jsonl"
        with out.open("w") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        hours = sum(float(e["duration"]) for e in rows) / 3600
        by_lang: dict[str, float] = {}
        for e in rows:
            by_lang[e.get("language", "?")] = by_lang.get(e.get("language", "?"), 0.0) + float(
                e["duration"]
            )
        print(
            f"wrote {out}: {len(rows):,} rows, {hours:.1f} h, "
            f"{ {k: round(v / 3600, 1) for k, v in by_lang.items()} }"
        )


if __name__ == "__main__":
    main()
