"""Convert aligned manifests to IPA, word by word, for the multilingual African model.

African words: africa-g2p's C library (native/, ctypes binding), one table per language — the
same code the device runs at inference. English: espeak-ng (en-us) + africa-g2p's English
normalisation, with leading/trailing punctuation split off and re-attached so English keeps
punctuation like the African tables do.

Each output row keeps the original spelling (transcript_orig, words[].orig) and replaces what
Pocket TTS reads (transcript, words[].word) with IPA. Rows that failed alignment are dropped.
Also counts letters left unconverted in a non-Latin script (a G2P table gap), per language.

Usage: python to_ipa.py IN.jsonl OUT.jsonl [--report report.json]
"""

import argparse
import json
import re
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from paths import AFRICA_G2P, DATA, RECIPE, REPO  # noqa: E402
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(AFRICA_G2P / "src"))
sys.path.insert(0, str(AFRICA_G2P / "native" / "python"))
from afrig2p import Afrig2p  # noqa: E402

from africa_g2p.english import _AFFRICATES, _NORMALISE, _backend  # noqa: E402

TABLES = (AFRICA_G2P / "native" / "data")
PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)
LATINISH = ("LATIN", "IPA", "MODIFIER", "GREEK", "COMBINING")


def unconverted(text: str) -> list[str]:
    """Letters from a non-Latin script still present after conversion (table gaps)."""
    out = []
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            name = unicodedata.name(ch, "")
            if not name.startswith(LATINISH) and "LETTER" in name:
                out.append(ch)
    return out


class English:
    def __init__(self) -> None:
        self.backend = _backend("en-us")
        self.cache: dict[str, str] = {}

    def batch(self, words: list[str]) -> None:
        todo = sorted({PUNCT.match(w).group(2) for w in words} - set(self.cache) - {""})
        if todo:
            out = self.backend.phonemize(todo, strip=True, njobs=8)
            for w, p in zip(todo, out):
                p = p.translate(_NORMALISE)
                for a, b in _AFFRICATES:
                    p = p.replace(a, b)
                self.cache[w] = p.replace(" ", "")  # one word in, one IPA word out

    def word(self, w: str) -> str:
        lead, core, trail = PUNCT.match(w).groups()
        return lead + (self.cache.get(core, "") if core else "") + trail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8")]
    rows = [r for r in rows if r.get("words")]
    eng = English()
    eng.batch([w["word"] for r in rows if r["g2p"] == "eng" for w in r["words"]])
    tables: dict[str, Afrig2p] = {}
    gaps: dict[str, Counter] = defaultdict(Counter)
    clips_with_gap: Counter = Counter()
    clips: Counter = Counter()
    with open(args.out, "w", encoding="utf-8") as fout:
        for r in rows:
            g = r["g2p"]
            clips[g] += 1
            if g != "eng" and g not in tables:
                tables[g] = Afrig2p(TABLES / f"{g}.afg2p")
            new_words = []
            for w in r["words"]:
                orig = w["word"]
                ipa = eng.word(orig) if g == "eng" else tables[g].ipa(orig)
                new_words.append({**w, "word": ipa, "orig": orig})
            r = dict(r)
            r["transcript_orig"] = r["transcript"]
            r["words"] = new_words
            r["transcript"] = " ".join(w["word"] for w in new_words)
            bad = unconverted(r["transcript"])
            if bad:
                clips_with_gap[g] += 1
                gaps[g].update(bad)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {g: {"clips": clips[g], "clips_with_unconverted": clips_with_gap[g],
                  "top_unconverted": gaps[g].most_common(15)} for g in clips}
    worst = sorted(((v["clips_with_unconverted"] / v["clips"], g) for g, v in report.items()
                    if v["clips_with_unconverted"]), reverse=True)
    print(f"{len(rows):,} rows -> {args.out}; languages with unconverted letters: {len(worst)}")
    for frac, g in worst[:15]:
        print(f"  {g}: {frac:.1%} of clips, e.g. {''.join(c for c, _ in report[g]['top_unconverted'][:12])}")
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
