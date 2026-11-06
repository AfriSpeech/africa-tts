"""Export a training checkpoint (EMA weights) to a pocket-tts model folder.

Writes OUT/model.safetensors, OUT/tokenizer.json and OUT/config.yaml, the config pointing at
hf://REPO/... so the folder can be uploaded to the Hugging Face repo REPO as is. The config is
the base model's (english_2026-04_24l) with the layer count of the checkpoint, and with Pocket
TTS's text clean-up turned off, because the model reads IPA:
  capitalize_first_letter: false   IPA has no capitals (training never capitalised)
  replace_characters: removed      ’ is an IPA mark (Amharic/Tigrinya ejectives), not a quote

Usage: python export_release.py TRAIN_CONFIG CKPT OUT_DIR HF_REPO
  e.g. export_release.py training/configs/finetune_africa_ipa.yaml \
           runs/africa_ipa_teacher/checkpoint_00100000.pt release/africa-tts AfriSpeech/africa-tts
"""

import argparse
import shutil
import sys
from pathlib import Path

import pocket_tts
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RECIPE, REPO

sys.path.insert(0, str(REPO))
from training.args import load_args
from training.checkpointing import export_pocket_safetensors
from training.modules.builders import build_models

BASE_CONFIG = Path(pocket_tts.__file__).parent / "config" / "english_2026-04_24l.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train_config")
    ap.add_argument("ckpt")
    ap.add_argument("out", type=Path)
    ap.add_argument("hf_repo")
    ap.add_argument("--tokenizer", default=str(RECIPE / "tokenizer" / "ipa_sp4000.json"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    args = load_args(a.train_config)
    model, mimi, _ = build_models(args)  # for a distillation config: the student
    payload = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    state = dict(payload["model"])
    state.update(payload.get("ema") or {})
    model.load_state_dict(state)
    export_pocket_safetensors(a.out / "model.safetensors", model.flow_lm, mimi)
    shutil.copy(a.tokenizer, a.out / "tokenizer.json")

    layers = len(model.flow_lm.transformer.layers)
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg.pop("weights_path_without_voice_cloning", None)
    cfg.pop("replace_characters", None)
    cfg["capitalize_first_letter"] = False
    cfg["weights_path"] = f"hf://{a.hf_repo}/model.safetensors"
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = f"hf://{a.hf_repo}/tokenizer.json"
    cfg["flow_lm"]["transformer"]["num_layers"] = layers
    (a.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    n = sum(v.numel() for v in model.flow_lm.state_dict().values()) / 1e6
    print(f"exported {a.out}: {layers} layers, {n:.0f}M flow-LM parameters")


if __name__ == "__main__":
    main()
