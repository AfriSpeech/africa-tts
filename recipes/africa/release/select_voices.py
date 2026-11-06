"""Pick one default voice prompt per training language from the aligned training data.

Prefers JW recordings (source "afrispeech"); languages without JW audio fall back to the other
sources (Tigrinya: africa-female-speech-v2; English: Ghanaian English). For each language, the
candidates are clips with confident alignment (mean word score) from which a 3.5-6 s prompt
can be cut: leading silence trimmed to 0.15 s, ending on a word boundary (+ 0.15 s tail).
Candidates are ranked by alignment confidence, then loudness sanity; the best one is written as
24 kHz mono WAV.

Usage: python select_voices.py TRAIN_JSONL OUT_DIR
Writes OUT_DIR/<code>.wav and OUT_DIR/voices.json (source clip, text, duration per language).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MAX_SEC, MIN_SEC, TAIL = 6.0, 3.5, 0.15
SOURCE_RANK = {"afrispeech": 0, "africa-female-speech-v2": 1, "ghana-english-tts-clean2": 2,
               "hifitts2": 3}


def cut(r: dict) -> tuple[float, float, list[dict]] | None:
    """(lead, end, words): skip leading silence (keep 0.15 s), end on a word boundary."""
    w = r["words"]
    if not w:
        return None
    lead = max(0.0, w[0]["start"] - 0.15)
    keep = [x for x in w if x["end"] + TAIL - lead <= MAX_SEC]
    if not keep:
        return None
    end = min(keep[-1]["end"] + TAIL, r["duration"])
    if end - lead < MIN_SEC:
        return None
    return lead, end, keep


def load(r: dict, lead: float, end: float) -> np.ndarray:
    sr = sf.info(r["path"]).samplerate
    a, _ = sf.read(r["path"], start=int((r["start"] + lead) * sr), frames=int((end - lead) * sr), dtype="float32",
                   always_2d=True)
    a = a.mean(1)
    if sr != 24000:
        g = np.gcd(sr, 24000)
        a = resample_poly(a, 24000 // g, sr // g).astype(np.float32)
    return a


def main() -> None:
    src, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cands: dict[str, list] = defaultdict(list)
    names = {}
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        g = r.get("g2p", "eng")
        names.setdefault(g, r.get("language", g))
        c = cut(r)
        if c is None:
            continue
        lead, end, keep = c
        conf = float(np.mean([x.get("score", 0.0) for x in keep]))
        worst = min(x.get("score", 0.0) for x in keep)
        if worst < -6:
            continue
        cands[g].append((SOURCE_RANK.get(r.get("source"), 9), -conf, r, lead, end, keep))
    meta = {}
    for g in sorted(names):
        best_src = min((c[0] for c in cands[g]), default=None)
        pool = sorted((c for c in cands[g] if c[0] == best_src), key=lambda c: c[1])[:40]
        chosen = None
        for _, negconf, r, lead, end, keep in pool:
            a = load(r, lead, end)
            rms = float(np.sqrt(np.mean(a**2)) + 1e-9)
            peak = float(np.abs(a).max())
            if 0.02 <= rms <= 0.3 and peak < 0.999:
                chosen = (r, lead, end, keep, a, -negconf)
                break
        if chosen is None:
            print(f"{g}: no usable clip ({len(cands[g])} candidates)")
            continue
        r, lead, end, keep, a, conf = chosen
        sf.write(out / f"{g}.wav", a, 24000, subtype="PCM_16")
        meta[g] = {"language": names[g], "source": r.get("source"), "subset": r.get("subset"),
                   "clip": f"{Path(r['path']).parent.name}/{Path(r['path']).name}@{r['start'] + lead:.2f}s",
                   "duration": round(end - lead, 2), "text": " ".join(x.get("orig", x["word"]) for x in keep),
                   "align_conf": round(conf, 3)}
    (out / "voices.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    by_src = defaultdict(int)
    for m in meta.values():
        by_src[m["source"]] += 1
    print(f"{len(meta)}/{len(names)} languages; by source: {dict(by_src)}")


if __name__ == "__main__":
    main()
