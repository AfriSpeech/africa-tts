import pytest
import tokenizers
import torch

from training.modules.builders import grow_text_table


def _tokenizer(pieces: list[str]) -> tokenizers.Tokenizer:
    return tokenizers.Tokenizer(tokenizers.models.Unigram([(p, -1.0) for p in pieces], 0, False))


def test_appended_pieces_copy_their_init_rows_and_padding_moves_to_the_end():
    old = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)  # 3 pieces + padding
    tok = _tokenizer(["<unk>", "e", "o", "ɛ", "ɔ"])
    grown = grow_text_table(old, 5, tok, {"ɛ": "e", "ɔ": "o"})
    assert grown.shape == (6, 4)
    assert torch.equal(grown[:3], old[:3])
    assert torch.equal(grown[3], old[1])  # ɛ <- e
    assert torch.equal(grown[4], old[2])  # ɔ <- o
    assert torch.equal(grown[5], old[3])  # padding


def test_a_list_of_pieces_starts_from_their_mean_row():
    old = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
    tok = _tokenizer(["<unk>", "e", "o", "eo"])
    grown = grow_text_table(old, 4, tok, {"eo": ["e", "o"]})
    assert torch.equal(grown[3], (old[1] + old[2]) / 2)
    assert torch.equal(grown[4], old[3])


def test_missing_init_is_an_error():
    old = torch.zeros(4, 4)
    tok = _tokenizer(["<unk>", "e", "o", "ɛ", "ɔ"])
    with pytest.raises(ValueError, match="ɔ"):
        grow_text_table(old, 5, tok, {"ɛ": "e"})


def test_tokenizer_that_does_not_extend_is_an_error():
    old = torch.zeros(4, 4)
    tok = _tokenizer(["<unk>", "e", "o"])
    with pytest.raises(ValueError, match="does not extend"):
        grow_text_table(old, 3, tok, {"ɛ": "e"})
