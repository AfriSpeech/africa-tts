import argparse
from pathlib import Path

import soundfile as sf
import torch
import yaml
from pocket_tts import TTSModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--voice", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="tag value, or a full <lang:...> token")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cfg", type=float, default=1.0, help="use 1.0 for CFG-baked students")
    ap.add_argument("--eos-threshold", type=float)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.cfg != 1.0:
        raise SystemExit("this helper expects a CFG-baked student; use recipes/africa/guidance.py for cfg != 1")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["weights_path"] = str(args.checkpoint)
    config["replace_characters"] = {}
    config["capitalize_first_letter"] = False
    config["flow_lm"]["lookup_table"]["tokenizer"] = "tokenizers"
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(args.tokenizer)
    config["flow_lm"]["lookup_table"]["n_bins"] = 4000
    runtime_config = args.checkpoint.with_name("lang_tags_inference_config.yaml")
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TTSModel.load_model(
        config=str(runtime_config), eos_threshold=args.eos_threshold
    ).to(device)
    state = model.get_state_for_audio_prompt(args.voice)
    prefix = args.tag if args.tag.startswith("<") else f"<lang:{args.tag}>"
    audio = model.generate_audio(state, f"{prefix} {args.text}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio.detach().cpu().numpy().astype("float32"), model.sample_rate)


if __name__ == "__main__":
    main()
