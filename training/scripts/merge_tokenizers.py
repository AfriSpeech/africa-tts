"""Merge a second language's Unigram tokenizer into a pretrained one, keeping every
pretrained id so the pretrained text table carries over (e.g. English + Twi).

Pieces of `add` that `base` lacks are appended with their own Unigram score plus
`--score-offset` (both are log-probabilities; the offset trades how often the new
pieces win on the second language against how often they change the segmentation of
the first). Writes:
  <out>                the merged tokenizer.json (set lookup_table.n_bins to its vocab size)
  <out>.init.json      {new piece: [base pieces]}: the base segmentation of each new
                       piece, for the training config's text_embedding_init_file (each
                       new row starts as the mean of those rows)

With --eval-base / --eval-add text files, prints tokens/word on each and the share of
base-language lines whose tokenization changed, for every --score-offset given; the
tokenizer is written with the first offset. --protect-base leaves out every new piece that
would change the tokenization of that base-language text (pruning is done at the first
offset), so the base language keeps its pretrained segmentation.

Usage:
  python -m training.scripts.merge_tokenizers BASE.json ADD.json out.json \
      --init-char ɛ=e --init-char ɔ=o --eval-base en.txt --eval-add twi.txt \
      --score-offset 0 --score-offset -2
"""

import json
import re
from pathlib import Path
from typing import Annotated, Any

import tokenizers
import typer

from pocket_tts.utils.utils import download_if_necessary

BYTE_PIECE = re.compile(r"^<0x[0-9A-F]{2}>$")


def _load(path: str) -> dict[str, Any]:
    data = json.loads(Path(download_if_necessary(path)).read_text())
    if data["model"]["type"] != "Unigram":
        raise SystemExit(f"{path}: only Unigram tokenizers are supported")
    return data


def _merged(
    base: dict[str, Any], add: dict[str, Any], offset: float, exclude: set[str] | None = None
) -> tuple[dict[str, Any], list[str]]:
    merged = json.loads(json.dumps(base))
    vocab = merged["model"]["vocab"]
    have = {p for p, _ in vocab}
    special = {t["content"] for t in add.get("added_tokens", [])}
    exclude = exclude or set()
    new = []
    for piece, score in add["model"]["vocab"]:
        if piece in have or piece in special or piece in exclude or BYTE_PIECE.match(piece):
            continue
        vocab.append([piece, float(score) + offset])
        have.add(piece)
        new.append(piece)
    return merged, new


def _tok(data: dict[str, Any]) -> tokenizers.Tokenizer:
    return tokenizers.Tokenizer.from_str(json.dumps(data))


def _lines(path: Path | None, limit: int = 20000) -> list[str]:
    if path is None:
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(line)
            if len(out) >= limit:
                break
    return out


def _protecting(
    base: dict[str, Any], add: dict[str, Any], offset: float, protect: list[str]
) -> set[str]:
    """New pieces to leave out so `protect` (base-language text) tokenizes exactly as
    with the base tokenizer: drop every new piece used on a line whose ids changed, and
    repeat (removing a piece can let another new piece win elsewhere) until none change."""
    ref = [e.ids for e in _tok(base).encode_batch(protect)]
    n_base = len(base["model"]["vocab"])
    exclude: set[str] = set()
    for round_ in range(1, 51):
        merged, new = _merged(base, add, offset, exclude)
        tok = _tok(merged)
        used: set[str] = set()
        changed = 0
        for e, r in zip(tok.encode_batch(protect), ref):
            if e.ids != r:
                changed += 1
                used.update(t for i in e.ids if i >= n_base and (t := tok.id_to_token(i)))
        print(f"  protect round {round_}: {changed} lines changed, dropping {len(used)} pieces")
        if not used:
            break
        exclude |= used
    return exclude


def _tokens_per_word(tok: tokenizers.Tokenizer, lines: list[str]) -> float:
    words = sum(len(x.split()) for x in lines)
    return sum(len(e.ids) for e in tok.encode_batch(lines)) / max(1, words)


def main(
    base: Annotated[str, typer.Argument(help="pretrained tokenizer.json (local or hf://)")],
    add: Annotated[str, typer.Argument(help="second language's tokenizer.json")],
    out: Annotated[Path, typer.Argument()],
    score_offset: Annotated[list[float], typer.Option()] = [0.0],  # noqa: B006
    init_char: Annotated[
        list[str],
        typer.Option(help="A=B: read letter A as B when segmenting a new piece for its init"),
    ] = [],  # noqa: B006
    eval_base: Annotated[Path | None, typer.Option(help="text in the base language")] = None,
    eval_add: Annotated[Path | None, typer.Option(help="text in the added language")] = None,
    protect_base: Annotated[
        Path | None,
        typer.Option(
            help="base-language text whose tokenization must not change: new pieces that "
            "would change it are left out (evaluate on different text with --eval-base)"
        ),
    ] = None,
) -> None:
    base_data, add_data = _load(base), _load(add)
    exclude: set[str] = set()
    if protect_base is not None:
        exclude = _protecting(
            base_data, add_data, score_offset[0], _lines(protect_base, limit=10**7)
        )
        print(f"protecting {protect_base}: {len(exclude)} new pieces left out")
    base_tok = _tok(base_data)
    en_lines, add_lines = _lines(eval_base), _lines(eval_add)
    en_ref = [e.ids for e in base_tok.encode_batch(en_lines)]
    if en_lines or add_lines:
        print(
            f"base tokenizer: base {_tokens_per_word(base_tok, en_lines):.3f} tok/word, "
            f"added language {_tokens_per_word(base_tok, add_lines):.3f} tok/word"
        )
    for offset in score_offset:
        merged, new = _merged(base_data, add_data, offset, exclude)
        tok = _tok(merged)
        if en_lines or add_lines:
            changed = sum(e.ids != r for e, r in zip(tok.encode_batch(en_lines), en_ref))
            print(
                f"offset {offset:+.1f}: +{len(new)} pieces | base "
                f"{_tokens_per_word(tok, en_lines):.3f} tok/word, "
                f"{changed / max(1, len(en_lines)):.1%} lines changed | added language "
                f"{_tokens_per_word(tok, add_lines):.3f} tok/word"
            )

    merged, new = _merged(base_data, add_data, score_offset[0], exclude)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False))

    # Each new row starts as the mean of the base rows its text segments into. The base
    # model segments it directly (no pre-tokenizer: "▁" is part of the piece); --init-char
    # maps letters the base vocab only covers through byte fallback to a near neighbour.
    char_map = dict(spec.split("=", 1) for spec in init_char)
    base_model = base_tok.model
    init: dict[str, list[str]] = {}
    for piece in new:
        text = "".join(char_map.get(c, c) for c in piece)
        parts = [t.value for t in base_model.tokenize(text)]
        kept = [p for p in parts if not BYTE_PIECE.match(p)]
        init[piece] = kept or parts
    Path(f"{out}.init.json").write_text(json.dumps(init, ensure_ascii=False, indent=0))
    print(
        f"wrote {out} (vocab {_tok(merged).get_vocab_size()}, offset {score_offset[0]:+.1f}) "
        f"and {out}.init.json ({len(init)} new pieces)"
    )


if __name__ == "__main__":
    typer.run(main)
