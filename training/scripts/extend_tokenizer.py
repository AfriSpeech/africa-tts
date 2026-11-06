"""Append pieces to a Unigram tokenizer.json (e.g. the released English one) for a new
language's letters, keeping every existing id so the pretrained text table carries over.

Each new piece gets the Unigram score of the piece its embedding will start from, and must
be paired with the same mapping in the training config's `text_embedding_init`; the
model config then needs `flow_lm.lookup_table.n_bins` = the new vocab size.

Example (Twi):
  uv run python -m training.scripts.extend_tokenizer \
      hf://kyutai/pocket-tts-without-voice-cloning/languages/english_2026-04_24l/tokenizer.json@00eac05ed3d16bdc3f6b5d598874019c34a89214 \
      data/tokenizer_en_twi.json --piece ɛ=e --piece ɔ=o
"""

import json
from pathlib import Path
from typing import Annotated

import tokenizers
import typer

from pocket_tts.utils.utils import download_if_necessary


def main(
    base: Annotated[str, typer.Argument(help="tokenizer.json to extend (local or hf://)")],
    out: Annotated[Path, typer.Argument(help="where to write the extended tokenizer.json")],
    piece: Annotated[
        list[str], typer.Option(help="NEW=INIT: append NEW, scored like existing piece INIT")
    ],
) -> None:
    data = json.loads(Path(download_if_necessary(base)).read_text())
    model = data["model"]
    if model["type"] != "Unigram":
        raise SystemExit(f"only Unigram tokenizers are supported, got {model['type']}")
    vocab: list[list[str | float]] = model["vocab"]
    scores: dict[str, float] = {str(p): float(s) for p, s in vocab}
    for spec in piece:
        new, sep, init = spec.partition("=")
        if not sep or not new or not init:
            raise SystemExit(f"--piece expects NEW=INIT, got {spec!r}")
        if new in scores:
            raise SystemExit(f"{new!r} is already in the vocabulary")
        if init not in scores:
            raise SystemExit(f"init piece {init!r} is not in the vocabulary")
        vocab.append([new, scores[init]])
        scores[new] = scores[init]
        print(f"appended {new!r} as id {len(vocab) - 1} (score of {init!r})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False))
    tok = tokenizers.Tokenizer.from_file(str(out))
    print(f"wrote {out}: vocab {tok.get_vocab_size()} (set n_bins to this)")


if __name__ == "__main__":
    typer.run(main)
