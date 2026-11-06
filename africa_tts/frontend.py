"""Text -> the IPA that Africa TTS was trained on.

African languages go through africa-g2p's rule tables; English through espeak-ng (en-us) with
africa-g2p's normalisation, word by word, keeping leading/trailing punctuation as training did.
Pinned to africa-g2p commit 76b53a8 (see pyproject.toml): the 0.3.2 release on PyPI predates
fixes to the Ghanaian tables and gives different IPA for Twi, Fante, Ewe, Igbo and Rundi.

The C front end in mobile/frontend produces byte-identical output for on-device use.
"""

import re

from africa_g2p import g2p

_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def to_ipa(text: str, lang: str) -> str:
    """Convert `text` in language `lang` (ISO 639-3 code, e.g. "twi", "eng") to model IPA."""
    if lang != "eng":
        return g2p(text, lang, output="ipa")
    words = []
    for w in text.split():
        lead, core, trail = _PUNCT.match(w).groups()
        ipa = g2p(core, "eng", output="ipa").replace(" ", "") if core else ""
        words.append(lead + ipa + trail)
    return " ".join(words)
