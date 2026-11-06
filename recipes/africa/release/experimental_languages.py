"""List the African languages Africa TTS was NOT trained on but can be tried on, because
africa-g2p has an IPA table for them: the model reads IPA, so any language whose sounds it has
learned from related languages may come out intelligible.

For every africa-g2p language with an IPA table and not in training:
  coverage   share of its phonemes (the IPA values of its grapheme table) whose symbols all
             occur in the training IPA: what the model has ever heard. Unseen symbols fall back
             to byte tokens the model never learned, so low coverage means some sounds will be
             wrong whatever the text.
  nearest    the trained language that sounds most alike, by phoneme-inventory overlap (Jaccard)
             plus a bonus for the same family (shared classification levels) and country. Its
             default voice is the suggested prompt.
  tier       A: coverage >= 0.98 and similarity >= 0.6;  B: coverage >= 0.9;  C: the rest.

Usage: python experimental_languages.py TRAIN_IPA_JSONL OUT_JSON [--md OUT_MD]
  africa-g2p rule files are read from $AFRICA_G2P_DIR/src/africa_g2p/languages.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import AFRICA_G2P, REPO  # noqa: E402

LANG_DIR = AFRICA_G2P / "src" / "africa_g2p" / "languages"
TRAINED = json.loads((REPO / "africa_tts" / "languages.json").read_text(encoding="utf-8"))
NAME2CC = {}


def country_code(c: str | None) -> str:
    if not c:
        return ""
    c = c.split(",")[0].split("/")[0].strip()
    if len(c) == 2 and c.isupper():
        return c
    if not NAME2CC:
        import pycountry

        for x in pycountry.countries:
            for n in {x.name, getattr(x, "common_name", x.name), getattr(x, "official_name", x.name)}:
                NAME2CC[n.lower()] = x.alpha_2
        NAME2CC.update({"ivory coast": "CI", "côte d'ivoire": "CI", "drc": "CD",
                        "democratic republic of the congo": "CD", "congo": "CG",
                        "tanzania": "TZ", "cape verde": "CV", "eswatini": "SZ"})
    return NAME2CC.get(c.lower(), "")


def family_levels(f: str | None) -> list[str]:
    if not f:
        return []
    parts = [p.strip().lower() for p in re.split(r"[,>/]", f) if p.strip()]
    return [p for p in parts if p not in {"niger-congo"}]  # often present in one format only


def phonemes(rule: dict) -> set[str]:
    out = set()
    for v in (rule.get("graphemes") or {}).values():
        if isinstance(v, str) and v.strip():
            out.add(unicodedata.normalize("NFC", v.strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train_ipa")
    ap.add_argument("out", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()

    seen = set()
    for line in open(a.train_ipa, encoding="utf-8"):
        seen.update(unicodedata.normalize("NFC", json.loads(line)["transcript"]))

    rules = {}
    for f in sorted(LANG_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        inv = phonemes(d)
        if inv:
            rules[d.get("code", f.stem)] = (d, inv)

    trained = {g: rules[g] for g in TRAINED if g in rules}
    out = {}
    for code, (d, inv) in rules.items():
        if code in TRAINED or code == "eng":
            continue
        covered = [p for p in inv if all(ch in seen or ch.isspace() for ch in p)]
        coverage = len(covered) / len(inv)
        fam, cc = family_levels(d.get("family")), country_code(d.get("country"))
        best = None
        for t, (td, tinv) in trained.items():
            jac = len(inv & tinv) / len(inv | tinv)
            tf = family_levels(td.get("family"))
            shared = len(set(fam) & set(tf))
            score = jac + 0.05 * min(shared, 4) + (0.1 if cc and cc == country_code(td.get("country")) else 0)
            if best is None or score > best[0]:
                best = (score, t, jac)
        score, nearest, jac = best
        tier = "A" if coverage >= 0.98 and score >= 0.6 else "B" if coverage >= 0.9 else "C"
        unseen = sorted({ch for p in inv for ch in p if ch not in seen and not ch.isspace()})
        out[code] = {"name": d.get("name", code), "country": cc or d.get("country", ""),
                     "family": d.get("family", ""), "nearest": nearest,
                     "nearest_name": TRAINED[nearest]["name"], "similarity": round(score, 3),
                     "phoneme_overlap": round(jac, 3), "coverage": round(coverage, 3),
                     "unseen_symbols": "".join(unseen), "tier": tier,
                     "g2p_confidence": d.get("confidence", "")}
    out = dict(sorted(out.items(), key=lambda kv: (kv[1]["tier"], -kv[1]["similarity"])))
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    tiers = {t: sum(1 for v in out.values() if v["tier"] == t) for t in "ABC"}
    print(f"{len(out)} experimental languages: tiers {tiers}")

    if a.md:
        rows = ["| Tier | Language | Code | Country | Closest trained language (suggested voice) | Similarity | IPA coverage | Unseen symbols |",
                "|:-:|---|---|---|---|--:|--:|---|"]
        for code, v in out.items():
            rows.append(f"| {v['tier']} | {v['name']} | `{code}` | {v['country']} | {v['nearest_name']} (`{v['nearest']}`) "
                        f"| {v['similarity']:.2f} | {v['coverage']:.0%} | {v['unseen_symbols'] or '—'} |")
        a.md.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
