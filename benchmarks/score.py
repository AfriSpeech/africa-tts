"""Score the benchmark: character error rate (CER) of Omnilingual ASR transcripts against the
input text, per language and system.

Text is normalised before scoring: Unicode NFC, lowercase, punctuation and symbols removed,
whitespace collapsed. A second CER also removes diacritics (combining marks), since the ASR often
drops tone marks even on real speech. CER is pooled over each language's sentences (total edits
/ total reference characters). Averages compare systems only over the languages they share.

Usage: python score.py BENCH_DIR [--names LANGS_JSON]
Writes BENCH_DIR/results.json and BENCH_DIR/results.md
"""

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer

ORDER = ["real", "turbo", "guided", "unguided", "mms", "omnivoice"]
TITLE = {"real": "Real speech", "turbo": "Africa TTS Turbo", "guided": "Africa TTS guided",
         "unguided": "Africa TTS unguided", "mms": "MMS-TTS", "omnivoice": "OmniVoice"}


def norm(t: str, strip_marks: bool = False) -> str:
    t = unicodedata.normalize("NFC", t).lower()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    if strip_marks:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
        t = unicodedata.normalize("NFC", t)
    return " ".join(t.split())


def cer(refs: list[str], hyps: list[str]) -> float:
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return float("nan")
    out = jiwer.process_characters([p[0] for p in pairs], [p[1] if p[1] else " " for p in pairs])
    return (out.substitutions + out.deletions + out.insertions) / sum(len(p[0]) for p in pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path)
    ap.add_argument("--names", type=Path, default=Path(__file__).resolve().parents[1] / "africa_tts" / "languages.json")
    a = ap.parse_args()
    names = {c: v["name"] for c, v in json.loads(a.names.read_text()).items()}
    rows = [json.loads(line) for line in open(a.bench / "testset.jsonl")]
    ref = {(r["lang"], r["id"]): r["text"] for r in rows}
    source = {r["lang"]: r["source"] for r in rows}
    systems = [s for s in ORDER if (a.bench / "asr" / f"{s}.jsonl").exists()]
    res: dict = defaultdict(dict)  # lang -> system -> metrics
    flags: dict = defaultdict(set)
    for s in systems:
        by_lang = defaultdict(list)
        for line in open(a.bench / "asr" / f"{s}.jsonl"):
            h = json.loads(line)
            by_lang[h["lang"]].append(h)
            if h.get("asr_lang") is None:
                flags[h["lang"]].add("no ASR language code")
        for lang, hs in by_lang.items():
            refs = [ref[(lang, h["id"])] for h in hs]
            hyp = [h["hyp"] for h in hs]
            res[lang][s] = {
                "cer": cer([norm(t) for t in refs], [norm(t) for t in hyp]),
                "cer_nodiac": cer([norm(t, True) for t in refs], [norm(t, True) for t in hyp]),
                "n": len(hs), "cut": sum(h["cut"] for h in hs),
            }
    langs = sorted(res, key=lambda c: names.get(c, c))
    out = {"systems": systems, "languages": {c: {"name": names.get(c, c), "source": source[c],
                                                  "flags": sorted(flags[c]), **res[c]} for c in langs}}

    # macro averages over shared languages, for each system pair with Turbo / each other
    def avg(sys_list, key="cer"):
        shared = [c for c in langs if all(s in res[c] for s in sys_list)]
        return shared, {s: sum(res[c][s][key] for c in shared) / len(shared) for s in sys_list} if shared else {}

    comps = []
    synth = [s for s in systems if s != "real"]
    for other in ("mms", "omnivoice"):
        if other in systems:
            group = [s for s in ("turbo", "guided", "unguided") if s in systems] + [other]
            shared, m = avg(group)
            comps.append({"vs": other, "languages": shared, "cer": m, "cer_nodiac": avg(group, "cer_nodiac")[1]})
    shared_all, m_all = avg(synth)
    out["averages"] = comps + [{"vs": "all", "languages": shared_all, "cer": m_all,
                                "cer_nodiac": avg(synth, "cer_nodiac")[1]}]
    own = [s for s in ("real", "turbo", "guided", "unguided") if s in systems]
    out["averages"].append({"vs": "real", "languages": avg(own)[0], "cer": avg(own)[1],
                            "cer_nodiac": avg(own, "cer_nodiac")[1]})
    (a.bench / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # markdown
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}"  # noqa: E731
    lines = ["| Language | Code | Text source | " + " | ".join(TITLE[s] for s in systems) + " |",
             "|---|---|---|" + "---:|" * len(systems)]
    for c in langs:
        cells = [pct(res[c][s]["cer"]) if s in res[c] else "—" for s in systems]
        note = " †" if flags[c] else ""
        lines.append(f"| {names.get(c, c)}{note} | `{c}` | {source[c]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Averages (mean of per-language CER, %, over the languages the systems share):")
    lines.append("")
    lines.append("| Comparison | Languages | " + " | ".join(TITLE[s] for s in systems) + " |")
    lines.append("|---|---:|" + "---:|" * len(systems))
    for comp in out["averages"]:
        label = {"mms": "vs MMS-TTS", "omnivoice": "vs OmniVoice", "all": "all systems",
                 "real": "vs real speech"}[comp["vs"]]
        cells = [pct(comp["cer"].get(s)) if s in comp["cer"] else "" for s in systems]
        lines.append(f"| {label} | {len(comp['languages'])} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Same, with diacritics removed before scoring:")
    lines.append("")
    lines.append("| Comparison | Languages | " + " | ".join(TITLE[s] for s in systems) + " |")
    lines.append("|---|---:|" + "---:|" * len(systems))
    for comp in out["averages"]:
        label = {"mms": "vs MMS-TTS", "omnivoice": "vs OmniVoice", "all": "all systems",
                 "real": "vs real speech"}[comp["vs"]]
        cells = [pct(comp["cer_nodiac"].get(s)) if s in comp["cer_nodiac"] else "" for s in systems]
        lines.append(f"| {label} | {len(comp['languages'])} | " + " | ".join(cells) + " |")
    (a.bench / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
