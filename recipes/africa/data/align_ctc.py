"""Word-align Pocket TTS manifests with ctc-forced-aligner (MahmoudAshraf/mms-300m-1130-forced-
aligner): one model for every language, transcripts romanized with uroman by the package.

Clips (<= 30 s, one window each, as the package does) are batched on the GPU with padding
masks; text processing, forced alignment, spans and word timestamps use the package's own
functions. Words that normalize to nothing (a stray dash or quote) are attached to the previous
word so the transcript text is kept. Output rows gain "words": [{word, start, end, score}] in
seconds from the clip start (what training/scripts/align_data.py writes, plus a score).

Usage: python align_ctc.py IN.jsonl OUT.jsonl [--batch-size 64]   (resumable: appends)
"""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
from ctc_forced_aligner import (get_alignments, get_spans, load_alignment_model,
                                postprocess_results, preprocess_text)
from scipy.signal import resample_poly

import ctc_forced_aligner.alignment_utils as _au
import torchaudio.functional as _taf


def _torchaudio_forced_align(log_probs, targets, input_lengths=None, target_lengths=None, blank=0):
    """Drop-in for the package's forced_align: its C++ port corrupts the heap on some inputs
    (crashes the process); torchaudio's original implementation validates them and raises."""
    paths, scores = _taf.forced_align(torch.from_numpy(log_probs), torch.from_numpy(targets).int(),
                                      blank=blank)
    return paths.numpy(), scores[0].numpy()  # 1-D per-frame scores, as the C++ version returns


_au.forced_align = _torchaudio_forced_align  # get_alignments looks it up in this module

SR = 16000
CHUNK = 4096  # rows read, length-sorted and batched at a time; written back in input order


def load(row: dict) -> np.ndarray | None:
    try:
        info = sf.info(row["path"])
        sr = info.samplerate
        wav, _ = sf.read(row["path"], start=int(round(row["start"] * sr)),
                         frames=int(round(row["duration"] * sr)), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != SR:
            g = math.gcd(sr, SR)
            wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
        return wav
    except Exception:  # noqa: BLE001
        return None


def word_units(text: str, lang: str) -> tuple[list[str], list[str]]:
    """(original words, romanized tokens) with empty-token words merged into the previous word."""
    _, text_starred = preprocess_text(text, romanize=True, language=lang)
    tokens_starred, _ = preprocess_text(text, romanize=True, language=lang)
    words = [t for t in text_starred if t != "<star>"]
    toks = [t for t in tokens_starred if t != "<star>"]
    out_w, out_t = [], []
    for w, t in zip(words, toks, strict=True):
        if t.strip():
            out_w.append(w)
            out_t.append(t)
        elif out_w:
            out_w[-1] += " " + w
    return out_w, out_t


def align_one(emission: torch.Tensor, row: dict, tokenizer, stride: float) -> dict | None:
    lang = "eng" if row.get("g2p") == "eng" else row.get("iso", "eng")
    words, toks = word_units(row["transcript"], lang)
    if not toks:
        return None
    tokens_starred, text_starred = [], []
    for w, t in zip(words, toks, strict=True):
        tokens_starred += ["<star>", t]
        text_starred += ["<star>", w]
    # CTC needs at least one frame per target token, plus one between equal neighbours; the
    # package's C++ forced_align does not check this and corrupts memory when it is violated.
    vocab = {k.lower() for k in tokenizer.get_vocab()} | {"<star>"}
    target = [c for c in " ".join(tokens_starred).split(" ") if c in vocab]
    needed = len(target) + sum(a == b for a, b in zip(target, target[1:]))
    if not target or needed > emission.size(0):
        return None
    try:
        segments, scores, blank = get_alignments(emission, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
        stamps = postprocess_results(text_starred, spans, stride, scores)
    except Exception:  # noqa: BLE001
        return None
    if len(stamps) != len(words):
        return None
    row = dict(row)
    row["words"] = [{"word": w, "start": round(s["start"], 3), "end": round(s["end"], 3),
                     "score": round(float(s["score"]), 3)} for w, s in zip(words, stamps)]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16, help="audio-loading threads")
    args = ap.parse_args()
    model, tokenizer = load_alignment_model("cuda", dtype=torch.float16)
    ratio = model.config.inputs_to_logits_ratio
    stride = ratio * 1000 / SR
    try:
        done = sum(1 for _ in open(args.out, encoding="utf-8"))
    except FileNotFoundError:
        done = 0
    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8")]
    print(f"{len(rows):,} rows, {done:,} already written", flush=True)
    pool = ThreadPoolExecutor(args.threads)
    n_ok = n_fail = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for c0 in range(done, len(rows), CHUNK):
            chunk = rows[c0:c0 + CHUNK]
            wavs = list(pool.map(load, chunk))
            order = sorted(range(len(chunk)), key=lambda i: len(wavs[i]) if wavs[i] is not None else 0)
            results: list[dict | None] = [None] * len(chunk)
            for b0 in range(0, len(order), args.batch_size):
                idx = [i for i in order[b0:b0 + args.batch_size] if wavs[i] is not None]
                if not idx:
                    continue
                lens = [len(wavs[i]) for i in idx]
                x = torch.zeros(len(idx), max(lens))
                for k, i in enumerate(idx):
                    x[k, :lens[k]] = torch.from_numpy(wavs[i])
                mask = (torch.arange(max(lens))[None] < torch.tensor(lens)[:, None]).long()
                with torch.inference_mode():
                    logits = model(x.cuda().half(), attention_mask=mask.cuda()).logits.float()
                frames = [int(model._get_feat_extract_output_lengths(n)) for n in lens]
                ems = []
                for k in range(len(idx)):
                    e = torch.log_softmax(logits[k, :frames[k]], dim=-1).cpu()
                    ems.append(torch.cat([e, torch.zeros(e.size(0), 1)], dim=1))  # <star> column
                # Sequential on purpose: the package's C++ forced_align is not thread-safe
                # (concurrent calls corrupt the heap); threads are only used for audio loading.
                for k, i in enumerate(idx):
                    results[i] = align_one(ems[k], chunk[i], tokenizer, stride)
            for i, r in enumerate(results):
                # Keep input order and one output line per input row (failed rows marked).
                if r is None:
                    n_fail += 1
                    fout.write(json.dumps({**chunk[i], "words": None}, ensure_ascii=False) + "\n")
                else:
                    n_ok += 1
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"{c0 + len(chunk):,}/{len(rows):,} | ok {n_ok:,} failed {n_fail:,}", flush=True)
    print(f"done: {n_ok:,} aligned, {n_fail:,} failed")


if __name__ == "__main__":
    main()
