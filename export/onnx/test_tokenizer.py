#!/usr/bin/env python3
"""Check that sherpa-onnx's SentencePiece re-implementation (sherpa-onnx/csrc/
sentence-piece-tokenizer.cc, re-implemented here byte for byte) gives the same token ids as the
model's own tokenizer.json, on real IPA sentences.

    python test_tokenizer.py --onnx-dir OUT --tokenizer tokenizer.json --jsonl valid.jsonl [--n 3000]
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

NEG_INF = float("-inf")


class SherpaSentencePiece:
    def __init__(self, vocab: dict, scores: dict):
        self.id2tok = {i: t for t, i in vocab.items()}
        self.trie = [{"next": {}, "id": -1, "score": 0.0}]
        for tok, i in vocab.items():
            node = 0
            for b in tok.encode("utf-8"):
                nxt = self.trie[node]["next"].get(b)
                if nxt is None:
                    nxt = len(self.trie)
                    self.trie.append({"next": {}, "id": -1, "score": 0.0})
                    self.trie[node]["next"][b] = nxt
                node = nxt
            self.trie[node]["id"] = i
            self.trie[node]["score"] = scores[tok]
        self.byte_id = [vocab.get(f"<0x{i:02X}>", -1) for i in range(256)]
        self.byte_score = [scores.get(f"<0x{i:02X}>", NEG_INF) for i in range(256)]

    def encode(self, text: str) -> list[int]:
        t = text.replace(" ", "▁")
        if not t.startswith("▁"):
            t = "▁" + t
        data = t.encode("utf-8")
        n = len(data)
        dp = [NEG_INF] * (n + 1)
        back = [-1] * (n + 1)
        back_id = [-1] * (n + 1)
        dp[n] = 0.0
        for i in range(n - 1, -1, -1):
            node = 0
            for j in range(i, n):
                nxt = self.trie[node]["next"].get(data[j])
                if nxt is None:
                    break
                node = nxt
                if self.trie[node]["id"] >= 0:
                    s = self.trie[node]["score"] + dp[j + 1]
                    if s > dp[i]:
                        dp[i], back[i], back_id[i] = s, j + 1, self.trie[node]["id"]
            if back[i] < 0:  # byte fallback
                b = data[i]
                dp[i] = self.byte_score[b] + dp[i + 1]
                back[i], back_id[i] = i + 1, self.byte_id[b]
        ids, i = [], 0
        while i < n:
            j, tid = back[i], back_id[i]
            if j <= i or tid < 0:
                break
            ids.append(tid)
            i = j
        return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    vocab = json.loads((a.onnx_dir / "vocab.json").read_text(encoding="utf-8"))
    scores = json.loads((a.onnx_dir / "token_scores.json").read_text(encoding="utf-8"))
    sherpa = SherpaSentencePiece(vocab, scores)
    hf = Tokenizer.from_file(str(a.tokenizer))
    texts = []
    for line in open(a.jsonl, encoding="utf-8"):
        texts.append(json.loads(line)["transcript"])
    step = max(1, len(texts) // a.n)
    texts = texts[::step][: a.n]
    bad = 0
    for t in texts:
        ref = hf.encode(t).ids
        got = sherpa.encode(t)
        if ref != got:
            bad += 1
            if bad <= 5:
                print("MISMATCH:", repr(t[:80]))
                print("  hf    :", [hf.id_to_token(i) for i in ref][:20])
                print("  sherpa:", [sherpa.id2tok[i] for i in got][:20])
    print(f"{len(texts) - bad}/{len(texts)} sentences identical ({bad} mismatches)")


if __name__ == "__main__":
    main()
