"""Assemble a self-contained on-device bundle for sherpa-onnx + the africa_tts front end.

    python mobile/make_bundle.py --onnx-dir out/africa-tts --g2p-dir g2p --voices-dir voices \
        --espeak-dir espeak-en --out bundle/africa-tts

Layout (what mobile/android/.../AfricaTts.kt and the Python example expect):
    lm_main.int8.onnx lm_flow.int8.onnx decoder.int8.onnx   int8: the bulk of the compute
    encoder.onnx text_conditioner.onnx                       fp32: int8 distorts the voice prompt
    vocab.json token_scores.json                             tokenizer
    g2p/<lang>.afg2p                                         africa-g2p tables (commit 76b53a8)
    voices/<lang>.wav voices/voices.json                     default voice per language
    espeak-ng-data/                                          English only (espeak-ng 1.51, GPL-3)
    README.md, LICENSE-espeak-ng.txt
"""

import argparse
import shutil
from pathlib import Path

MIXED = {
    "lm_main.int8.onnx": "lm_main.int8.onnx",
    "lm_flow.int8.onnx": "lm_flow.int8.onnx",
    "decoder.int8.onnx": "decoder.int8.onnx",
    "encoder.onnx": "encoder.onnx",
    "text_conditioner.onnx": "text_conditioner.onnx",
    "vocab.json": "vocab.json",
    "token_scores.json": "token_scores.json",
}

README = """# {name}: sherpa-onnx bundle

On-device files for [{name}](https://huggingface.co/{repo}), for
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)'s Pocket TTS backend plus the africa_tts
front end (https://github.com/AfriSpeech/africa-tts/tree/main/mobile).

The model reads IPA: convert text with the front end (`g2p/` tables; English needs
`espeak-ng-data/` and libespeak-ng **1.51**), then pass the IPA to sherpa-onnx with a reference
voice (`voices/<lang>.wav`) and `temperature` "0.3", `num_steps` 1, `silence_scale` 1.0.

Precision: int8 lm_main, lm_flow and decoder; fp32 encoder and text_conditioner.
License: model CC-BY-NC-4.0. espeak-ng-data: GPL-3.0-or-later (LICENSE-espeak-ng.txt).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--g2p-dir", type=Path, required=True)
    ap.add_argument("--voices-dir", type=Path, required=True)
    ap.add_argument("--espeak-dir", type=Path, help="folder containing espeak-ng-data (English)")
    ap.add_argument("--espeak-license", type=Path, default=Path("/usr/share/common-licenses/GPL-3"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="Africa TTS")
    ap.add_argument("--repo", default="AfriSpeech/africa-tts")
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    for src, dst in MIXED.items():
        shutil.copy(a.onnx_dir / src, a.out / dst)
    shutil.copytree(a.g2p_dir, a.out / "g2p")
    shutil.copytree(a.voices_dir, a.out / "voices")
    if a.espeak_dir:
        shutil.copytree(a.espeak_dir / "espeak-ng-data", a.out / "espeak-ng-data")
        if a.espeak_license.exists():
            shutil.copy(a.espeak_license, a.out / "LICENSE-espeak-ng.txt")
    (a.out / "README.md").write_text(README.format(name=a.name, repo=a.repo))
    total = sum(p.stat().st_size for p in a.out.rglob("*") if p.is_file()) / 1e6
    print(f"bundle {a.out}: {total:.0f} MB")


if __name__ == "__main__":
    main()
