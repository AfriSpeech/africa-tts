"""Train the shared IPA SentencePiece tokenizer and compare vocabulary sizes.

Corpus: IPA transcripts from the training manifests, sampled per language with weight
proportional to sqrt(hours) (small languages are repeated, large ones subsampled) so no single
language dominates the pieces. Spec = Kyutai's released tokenizers (training/scripts/
train_tokenizer.py) with character coverage 1.0, so every IPA symbol gets a piece.
Evaluation: tokens per word on the validation manifests, overall and per language.

Usage: python build_ipa_tokenizer.py OUT_DIR --train a.jsonl b.jsonl --valid c.jsonl d.jsonl
       [--vocab 4000 --vocab 8000 --char] [--lines 2000000]
"""

import argparse
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def load(paths: list[str]) -> dict[str, list[tuple[str, float]]]:
    by: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("transcript"):
                by[r["g2p"]].append((r["transcript"], r["duration"]))
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--valid", nargs="+", required=True)
    ap.add_argument("--vocab", type=int, action="append", default=[])
    ap.add_argument("--char", action="store_true", help="also train a character-level model")
    ap.add_argument("--lines", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train = load(args.train)
    hours = {g: sum(d for _, d in v) / 3600 for g, v in train.items()}
    weight = {g: math.sqrt(h) for g, h in hours.items()}
    total_w = sum(weight.values())
    corpus = args.out / "ipa_corpus.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for g, rows in train.items():
            n = max(200, int(args.lines * weight[g] / total_w))
            picks = [rows[rng.randrange(len(rows))][0] for _ in range(n)] if n > len(rows) \
                else [t for t, _ in rng.sample(rows, n)]
            for t in picks:
                f.write(t + "\n")
    print(f"corpus: {corpus} ({sum(1 for _ in open(corpus)):,} lines, {len(train)} languages)")

    specs = [(f"sp{v}", ["--vocab-size", str(v), "--character-coverage", "1.0"]) for v in args.vocab]
    if args.char:
        specs.append(("char", ["--model-type", "char", "--vocab-size", "2000",
                               "--character-coverage", "1.0"]))
    valid = load(args.valid)
    import sentencepiece as spm

    for name, extra in specs:
        prefix = str(args.out / f"ipa_{name}")
        cmd = [sys.executable, "-m", "training.scripts.train_tokenizer", prefix, str(corpus), *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{name}: training failed\n{r.stderr[-1500:]}")
            continue
        sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
        per = {}
        tw = tt = 0
        for g, rows in valid.items():
            texts = [t for t, _ in rows]
            n_tok = sum(len(sp.encode(t)) for t in texts)
            n_w = sum(len(t.split()) for t in texts)
            per[g] = n_tok / max(1, n_w)
            tw += n_w
            tt += n_tok
        worst = sorted(per.items(), key=lambda kv: -kv[1])[:5]
        print(f"{name}: vocab {sp.get_piece_size()} | tokens/word overall {tt / tw:.2f} | "
              f"median language {sorted(per.values())[len(per) // 2]:.2f} | worst "
              + ", ".join(f"{g} {v:.2f}" for g, v in worst), flush=True)


if __name__ == "__main__":
    main()
