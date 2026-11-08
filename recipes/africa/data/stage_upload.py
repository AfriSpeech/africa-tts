"""Stage the private AfriSpeech dataset: audio (hard links, no copy) + relative manifests.

Layout:
  audio/<source>/<subset>/<file>.flac     only the files the kept clips point into
  train.jsonl, valid.jsonl                 one row per clip; "path" relative to the repo root
  README.md                                written separately

Usage: python stage_upload.py STAGE_DIR
"""

import json
import os
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from paths import DATA  # noqa: E402
from collections import Counter
from pathlib import Path

IPA = (DATA / "africa-tts/ipa")
ROOTS = {  # absolute audio prefix -> folder name in the dataset
    str(DATA / "afrispeech-jw/audio/") + "/": "audio/jw/",
    str(DATA / "female-v2-tts/audio/") + "/": "audio/africa-female-speech-v2/",
    str(DATA / "ghana-english-tts/audio/") + "/": "audio/ghana-english/",
    str(DATA / "hifitts2/audio/hifitts2_audio/") + "/": "audio/hifitts2/",
}


def main(stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    linked: set[str] = set()
    for split, src in (("train", "train_final.jsonl"), ("valid", "valid_final.jsonl")):
        n = 0
        hours = Counter()
        with open(IPA / src, encoding="utf-8") as fin, open(stage / f"{split}.jsonl", "w",
                                                              encoding="utf-8") as fout:
            for line in fin:
                r = json.loads(line)
                prefix = next(p for p in ROOTS if r["path"].startswith(p))
                rel = ROOTS[prefix] + r["path"][len(prefix):]
                if r["path"] not in linked:
                    dst = stage / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        os.link(r["path"], dst)
                    linked.add(r["path"])
                r["path"] = rel
                r.pop("align", None)
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                hours[r["source"]] += r["duration"] / 3600
        print(f"{split}: {n:,} clips | " + ", ".join(f"{k} {v:.1f} h" for k, v in hours.items()))
    total = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file() and f.suffix in (".flac", ".mp3", ".wav"))
    print(f"audio files linked: {len(linked):,}, {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
