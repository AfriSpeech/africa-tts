#!/usr/bin/env bash
# Finish the Twi pipeline on all Twi in the converted episodes (~350h) (conversion stopped 2026-11-05):
# keep only converted episodes -> MMS alignment -> merge -> teacher training with the
# merged English+Twi tokenizer (data/tok/en_twi_4000.json). Text stays real Twi.
set -e
cd /mnt/volume_d2wey28/projects/pocket-tts-twi
export HF_HOME=/mnt/volume_d2wey28/hf_cache
source .venv/bin/activate

echo "=== [1/4] Manifests restricted to converted audio ==="
python - <<'PY'
import json
from pathlib import Path

root = Path("data/twi/manifests")  # full extraction lists: all Twi chunks in each episode
vc_dir = Path("/mnt/volume_d2wey28/projects/pocket-tts-twi/data/twi/audio_vc")
for split in ["train_twi", "valid_twi"]:
    n = 0
    h = 0.0
    with (root / f"{split}.jsonl").open() as fin, (root / f"{split}_vc.jsonl").open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            vc = vc_dir / Path(r["path"]).name
            if vc.exists():
                r["path"] = str(vc)
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                h += r["duration"]
    print(f"{split}_vc.jsonl: {n:,} rows, {h / 3600:.1f} h")
PY

echo "=== [2/4] MMS 1B CTC alignment (aka) ==="
for split in train valid; do
    python -m training.scripts.align_data \
        data/twi/manifests/${split}_twi_vc.jsonl \
        data/${split}_twi_aligned.jsonl \
        --model facebook/mms-1b-all \
        --lang aka \
        --batch-size 32 --resume
done

echo "=== [3/4] Merge + drop rows without a usable word cut ==="
python data/merge_aligned.py --out-dir data

echo "=== [4/4] Teacher training ==="
python training/train.py training/configs/finetune_twi.yaml
