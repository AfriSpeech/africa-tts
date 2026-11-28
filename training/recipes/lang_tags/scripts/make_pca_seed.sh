set -euo pipefail

TEACHER_ARGS=${1:?usage: make_pca_seed.sh TEACHER_ARGS TEACHER_CKPT OUT_SEED}
TEACHER_CKPT=${2:?}
OUT_SEED=${3:?}

PY=${PY:-python}
AFRICA_TTS=${AFRICA_TTS:-$(cd "$(dirname "$0")/../../../.." && pwd)}
BASE_CONFIG=${BASE_CONFIG:-pocket_tts/config/english_2026-04_24l.yaml}
TOKENIZER=${TOKENIZER:?set TOKENIZER to the converted tokenizer json}
COV=${COV:-${OUT_SEED%.pt}_cov.pt}
BATCHES=${BATCHES:-32}
LAYERS=${LAYERS:-8}
D_MODEL=${D_MODEL:-512}
NUM_HEADS=${NUM_HEADS:-8}
HIDDEN_SCALE=${HIDDEN_SCALE:-1}
LOOKUP_DIM=${LOOKUP_DIM:-512}
Z_DIM=${Z_DIM:-1024}
NBINS=${NBINS:-4000}

read -r -a extra_overrides <<< "${EXTRA_OVERRIDES:-}"

cd "$AFRICA_TTS"
"$PY" -m training.scripts.residual_cov \
  "$TEACHER_ARGS" "$TEACHER_CKPT" "$COV" --batches "$BATCHES" --ema
"$PY" -m training.scripts.shrink_checkpoint "$TEACHER_CKPT" --ema \
  --config "$BASE_CONFIG" \
  --override "flow_lm.transformer.num_layers=$LAYERS" \
  --override "flow_lm.transformer.d_model=$D_MODEL" \
  --override "flow_lm.transformer.num_heads=$NUM_HEADS" \
  --override "flow_lm.transformer.hidden_scale=$HIDDEN_SCALE" \
  --override "flow_lm.lookup_table.dim=$LOOKUP_DIM" \
  --override "flow_lm.z_dim=$Z_DIM" \
  --override "flow_lm.lookup_table.tokenizer=tokenizers" \
  --override "flow_lm.lookup_table.tokenizer_path=$TOKENIZER" \
  --override "flow_lm.lookup_table.n_bins=$NBINS" \
  "${extra_overrides[@]}" \
  --cov "$COV" --out "$OUT_SEED"
