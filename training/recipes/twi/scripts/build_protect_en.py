"""English text whose tokenization the merged tokenizer must not change (pruning set).
Held-out English for checking stays in data/tok/eval_en.txt (podcast valid + README)."""

import csv
import json
import random
import sys

from huggingface_hub import hf_hub_download

csv.field_size_limit(sys.maxsize)
lines: list[str] = []
for repo, fname in [
    ("AfriSpeech/africa-corpus", "english_cache.csv"),
    ("AfriSpeech/african-corpus-jw", "english_cache.csv"),
]:
    path = hf_hub_download(repo, fname, repo_type="dataset")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = next(c for c in ("eng", "text", "local", "english") if c in (reader.fieldnames or []))
        rows = [(r.get(col) or "").strip() for r in reader]
    rows = [r.replace("\n", " ") for r in rows if r]
    random.Random(0).shuffle(rows)
    lines += rows[:150_000]
    print(repo, fname, "column", col, "rows", len(rows))
for line in open("data/twi/manifests/train_english.jsonl"):
    t = json.loads(line)["transcript"].strip()
    if t:
        lines.append(t)
with open("data/tok/protect_en.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")
print("protect_en.txt:", len(lines), "lines,", sum(len(x.split()) for x in lines), "words")
