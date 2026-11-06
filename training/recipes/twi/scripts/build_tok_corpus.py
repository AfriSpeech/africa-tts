"""Twi tokenizer corpus (pristine-twi, cased, sentence-split + our Twi train transcripts)
and evaluation text (our Twi valid transcripts; English podcast transcripts + repo docs)."""

import glob
import json
import re
from pathlib import Path

out = Path("data/tok")
out.mkdir(exist_ok=True)
split = re.compile(r"(?<=[.!?])\s+")
n = chars = 0
with open(out / "twi_corpus.txt", "w") as f:
    for fn in sorted(glob.glob("data/pristine-twi/data/*.parquet")):
        import pyarrow.parquet as pq

        for doc in pq.read_table(fn, columns=["twi"]).column("twi").to_pylist():
            for para in (doc or "").split("\n"):
                for s in split.split(para.strip()):
                    s = s.strip()
                    if len(s) >= 3:
                        f.write(s + "\n")
                        n += 1
                        chars += len(s)
    for line in open("data/twi/manifests/train_twi.jsonl"):
        t = json.loads(line)["transcript"].strip()
        if t:
            f.write(t + "\n")
            n += 1
            chars += len(t)
print(f"twi corpus: {n:,} lines, {chars / 1e6:.0f}M chars")

with open(out / "eval_twi.txt", "w") as f:
    for line in open("data/twi/manifests/valid_twi.jsonl"):
        f.write(json.loads(line)["transcript"].strip() + "\n")
with open(out / "eval_en.txt", "w") as f:
    for line in open("data/twi/manifests/valid_english.jsonl"):
        f.write(json.loads(line)["transcript"].strip() + "\n")
    for line in open(out / "en_readme.md"):
        line = re.sub(r"[`#*\[\]()|>]", " ", line).strip()
        if len(line.split()) >= 5 and "http" not in line:
            f.write(line + "\n")
for p in ["eval_twi.txt", "eval_en.txt"]:
    print(p, sum(1 for _ in open(out / p)), "lines")
