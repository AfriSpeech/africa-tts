set -euo pipefail

OUT_DIR=${1:?usage: build_tokenizer.sh OUT_DIR TRAIN VALID TAG [TAG ...]}
TRAIN=${2:?}
VALID=${3:?}
shift 3
if (($# == 0)); then
  echo "provide at least one language tag" >&2
  exit 2
fi

PY=${PY:-python}
AFRICA_TTS=${AFRICA_TTS:-$(cd "$(dirname "$0")/../../../.." && pwd)}
VOCAB=${VOCAB:-4000}
LINES=${LINES:-2000000}

mkdir -p "$OUT_DIR"
cd "$AFRICA_TTS"
"$PY" recipes/africa/data/build_ipa_tokenizer.py "$OUT_DIR" \
  --train "$TRAIN" \
  --valid "$VALID" \
  --vocab "$VOCAB" \
  --lines "$LINES" \
  --user-defined-symbols "$@"
"$PY" -m training.scripts.convert_tokenizer "$OUT_DIR/ipa_sp4000.model" \
  --out "$OUT_DIR/ipa_sp4000.json" \
  --texts "$VALID"
