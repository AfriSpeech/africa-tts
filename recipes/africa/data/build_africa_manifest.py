"""Combined IPA-only training manifest for the multilingual African model.

Keeps rows whose language has an africa-g2p IPA table (English goes through espeak-ng),
tags each row with:
  g2p    the africa-g2p code used for IPA (macro codes mapped: swa->swh, aka->twi, ...)
  align  the MMS-1b-all adapter used for alignment, or "universal" when MMS has none
and writes OUT/{train,valid}_all.jsonl plus per-aligner shards OUT/align/<align>_{train,valid}.jsonl.

Usage: python build_africa_manifest.py OUT
"""

import glob
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from paths import AFRICA_G2P, DATA, RECIPE, REPO  # noqa: E402
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(AFRICA_G2P / "native" / "python"))
from afrig2p import Afrig2p  # noqa: E402

G2P_DATA = (AFRICA_G2P / "native" / "data")
MACRO = {"swa": "swh", "orm": "gaz", "ful": "fuv", "aka": "twi", "kon": "kng", "mlg": "plt",
         "luy": "lwg", "kln": "niq"}
# MMS adapter codes that differ from the data's ISO code.
MMS_ALIAS = {"twi": "aka", "swh": "swh", "swa": "swh"}
SOURCES = [
    str((DATA / "afrispeech-jw/manifests/*.jsonl")),
    str((DATA / "female-v2-tts/manifests/*.jsonl")),
    str((DATA / "ghana-english-tts/manifests/*.jsonl")),
]


def g2p_code(code: str) -> str | None:
    if code == "eng":
        return "eng"
    for c in (code, MACRO.get(code)):
        p = G2P_DATA / f"{c}.afg2p" if c else None
        if p and p.exists():
            try:
                if Afrig2p(p).ipa("a") is not None:
                    return c
            except Exception:  # noqa: BLE001
                pass
    return None


def main(out: Path) -> None:
    adapters = {f.split(".")[1] for f in HfApi().list_repo_files("facebook/mms-1b-all")
                if f.startswith("adapter.")}
    cache: dict[str, str | None] = {}
    rows = defaultdict(list)
    hours = defaultdict(float)
    for pattern in SOURCES:
        for f in glob.glob(pattern):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                iso = r["iso"]
                if iso not in cache:
                    cache[iso] = g2p_code(iso)
                g = cache[iso]
                if g is None:
                    continue
                a = next((c for c in (MMS_ALIAS.get(iso), iso, MMS_ALIAS.get(g), g) if c in adapters),
                         "universal")
                r["g2p"], r["align"] = g, a
                split = "valid" if r.get("split") == "val" else "train"
                rows[(a, split)].append(r)
                hours[a] += r["duration"] / 3600
    (out / "align").mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        with open(out / f"{split}_all.jsonl", "w", encoding="utf-8") as fa:
            for (a, s), rs in sorted(rows.items()):
                if s != split:
                    continue
                with open(out / "align" / f"{a}_{split}.jsonl", "w", encoding="utf-8") as fs:
                    for r in rs:
                        line = json.dumps(r, ensure_ascii=False) + "\n"
                        fs.write(line)
                        fa.write(line)
    uni = hours.get("universal", 0.0)
    print(f"aligners: {len(hours)} ({len(hours) - (1 if uni else 0)} MMS adapters + universal)")
    print(f"hours: MMS adapters {sum(hours.values()) - uni:.1f} h | universal aligner {uni:.1f} h | "
          f"total {sum(hours.values()):.1f} h")
    langs_uni = sorted({r["iso"] for (a, _), rs in rows.items() if a == "universal" for r in rs})
    print("languages needing the universal aligner:", langs_uni)
    n_valid = sum(len(rs) for (a, s), rs in rows.items() if s == "valid")
    print(f"validation clips: {n_valid:,}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
