"""Language lists (no PyTorch needed)."""

import json
from pathlib import Path

_HERE = Path(__file__).parent
LANGS = json.loads((_HERE / "languages.json").read_text(encoding="utf-8"))
EXPERIMENTAL = json.loads((_HERE / "languages_experimental.json").read_text(encoding="utf-8"))


def languages() -> dict[str, dict]:
    """{code: {"name", "hours"}} for every language the models were trained on."""
    return dict(LANGS)


def experimental_languages() -> dict[str, dict]:
    """Languages the models were NOT trained on but that africa-g2p has an IPA table for:
    {code: {"name", "country", "nearest" (closest trained language, whose voice is used by
    default), "coverage" (share of its phonemes seen in training), "tier" (A/B/C), ...}}.
    Quality is unknown: please report how they sound (see docs/EXPERIMENTAL_LANGUAGES.md)."""
    return dict(EXPERIMENTAL)


def voice_language(lang: str) -> str:
    """The trained language whose default voice serves `lang` (itself, or the nearest one)."""
    if lang in EXPERIMENTAL:
        return EXPERIMENTAL[lang]["nearest"]
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}")
    return lang
