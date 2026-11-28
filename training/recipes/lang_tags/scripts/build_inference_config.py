import argparse
from pathlib import Path

import yaml


def dotted_set(config, key, value):
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    resolved = yaml.safe_load((args.run_dir / "args.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(resolved["model_config"]).read_text(encoding="utf-8"))
    for key, value in resolved.get("model_overrides", {}).items():
        dotted_set(config, key, value)
    config["weights_path"] = str(args.checkpoint)
    config["replace_characters"] = {}
    config["capitalize_first_letter"] = False
    config["flow_lm"]["lookup_table"]["tokenizer"] = "tokenizers"
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(args.tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
