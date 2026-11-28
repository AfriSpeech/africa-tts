import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as fin:
            for line in fin:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag-field", default="language")
    ap.add_argument("--condition-field", default="condition_prefix")
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--tag-prefix", default="<lang:{tag}>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    allowed = set(args.tags) if args.tags else None
    counts = defaultdict(int)
    hours = defaultdict(float)
    skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fout:
        for row in load_rows(args.inputs):
            tag = str(row.get(args.tag_field, "") or "")
            if not tag or (allowed is not None and tag not in allowed):
                skipped += 1
                continue
            transcript = str(row.get("transcript", "") or "").strip()
            if not transcript or not row.get("path") or float(row.get("duration", 0) or 0) <= 0:
                skipped += 1
                continue
            if not args.overwrite and row.get(args.condition_field):
                prefix = str(row[args.condition_field])
            else:
                prefix = args.tag_prefix.format(tag=tag)
            row[args.condition_field] = prefix
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[tag] += 1
            hours[tag] += float(row.get("duration", 0) or 0) / 3600.0
    for tag in sorted(counts):
        print(f"{tag}: {counts[tag]:,} rows / {hours[tag]:.2f}h")
    print(f"wrote {sum(counts.values()):,} rows to {args.out}; skipped {skipped:,}")


if __name__ == "__main__":
    main()
