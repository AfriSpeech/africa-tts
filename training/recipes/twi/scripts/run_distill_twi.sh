#!/usr/bin/env bash
# Distill the 24-layer Twi teacher (runs/twi_500h) into the 6-layer fast model.
#
#   ./data/run_distill_twi.sh          # run now (teacher checkpoint must exist)
#   ./data/run_distill_twi.sh --wait   # wait for the teacher run to finish, then run
#
# Log: data/distill_twi.log when started via nohup (see bottom of this file).
set -euo pipefail
cd /mnt/volume_d2wey28/projects/pocket-tts-twi

CONFIG=training/configs/depth_distill_twi.yaml
TEACHER_CKPT=runs/twi_500h/checkpoint_00200000.pt

if [[ "${1:-}" == "--wait" ]]; then
    echo "[$(date '+%F %T')] waiting for $TEACHER_CKPT and the teacher train.py to exit..."
    # The checkpoint appears before train.py finishes writing model.safetensors and
    # exits; wait for both so the two runs never share the GPU.
    until [[ -f "$TEACHER_CKPT" ]] && ! pgrep -f "training/train.py training/configs/finetune_twi.yaml" >/dev/null; do
        sleep 300
    done
    echo "[$(date '+%F %T')] teacher finished."
fi

if [[ ! -f "$TEACHER_CKPT" ]]; then
    echo "teacher checkpoint $TEACHER_CKPT not found; is runs/twi_500h finished?" >&2
    exit 1
fi
if pgrep -f "training/train.py training/configs/finetune_twi.yaml" >/dev/null; then
    echo "teacher training is still running; use --wait" >&2
    exit 1
fi

echo "[$(date '+%F %T')] === Depth distillation 24L -> 6L (Twi) ==="
uv run training/train.py "$CONFIG"
echo "[$(date '+%F %T')] done: runs/twi_500h_distill_6l/model.safetensors"

# Start detached:
#   nohup ./data/run_distill_twi.sh --wait > data/distill_twi.log 2>&1 &
