"""Samples for the Space, made exactly the way users run the models: the africa_tts package
(plain pocket-tts inference, no classifier-free guidance, temperature 0.3) with each language's
bundled default voice.

Texts come from a meta.json written by synth_udhr.py (UDHR Articles 1, 3, 5 per language). Each
call adds one model's audio under LABEL; run once per model.

Usage: python synth_samples.py META_IN OUT_DIR --label teacher --model release/africa-tts [--device cuda]
Writes OUT_DIR/audio/<lang>_art<n>_<label>.wav, OUT_DIR/audio/<lang>_voice.wav, OUT_DIR/meta.json
"""

import argparse
import json
import shutil
from pathlib import Path

import scipy.io.wavfile
from pocket_tts.utils.utils import download_if_necessary

from africa_tts import AfricaTTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("meta_in", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True, help="africa-tts | africa-tts-turbo | local folder")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    (a.out / "audio").mkdir(parents=True, exist_ok=True)
    meta_out = a.out / "meta.json"
    if meta_out.exists():
        meta = json.loads(meta_out.read_text())
    else:
        src = json.loads(a.meta_in.read_text())
        items = [{k: it[k] for k in ("g2p", "language", "article", "text", "ipa")} | {"synth": {}}
                 for it in src["items"]]
        meta = {"labels": [], "items": items}

    tts = AfricaTTS.load(a.model)
    tts.model.to(a.device)
    for i, it in enumerate(meta["items"]):
        g = it["g2p"]
        voice = tts.default_voice(g)
        it["voice"] = f"{g}_voice.wav"
        if not (a.out / "audio" / it["voice"]).exists():
            shutil.copy(download_if_necessary(voice), a.out / "audio" / it["voice"])
        audio = tts.generate(it["ipa"], g, ipa=True)  # same IPA as the page shows
        fname = f"{g}_art{it['article']}_{a.label}.wav"
        scipy.io.wavfile.write(a.out / "audio" / fname, tts.sample_rate, audio)
        it["synth"][a.label] = fname
        if i % 20 == 0:
            print(f"{i + 1}/{len(meta['items'])}", flush=True)
    if a.label not in meta["labels"]:
        meta["labels"].append(a.label)
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"{len(meta['items'])} samples ({a.label}) -> {a.out}")


if __name__ == "__main__":
    main()
