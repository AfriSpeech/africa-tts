"""Text -> the IPA that Africa TTS was trained on.

African languages go through africa-g2p's rule tables; English through espeak-ng (en-us).
Supports automatic English code-switching for loanwords, nouns and named entities
using greedy longest-match phrase substitution (favouring longer phrases up to 5 words
before shorter individual words).
"""

import gzip
import re
from functools import lru_cache
from pathlib import Path

from africa_g2p import g2p

_DATA_DIR = Path(__file__).resolve().parent / "data"
_LEXICON_PATH = _DATA_DIR / "codeswitch_en.tsv.gz"
_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)

# Greek/Cyrillic lookalikes that keyboards and websites (and FLORES) use for African Latin
# letters, e.g. Twi "ԑ" (Cyrillic) or Ewe "ͻ" (Greek) for ɛ and ɔ. The G2P tables only know the
# Latin letters. Kept in sync with LOOKALIKES in mobile/frontend/africa_tts_frontend.c.
LOOKALIKES = {
    "ͻ": "ɔ", "Ͻ": "Ɔ", "ԑ": "ɛ", "Ԑ": "Ɛ", "ε": "ɛ", "Ε": "Ɛ", "Γ": "Ɣ", "γ": "ɣ",
    "ῖ": "ĩ",
}
_LOOKALIKES = str.maketrans(LOOKALIKES)


@lru_cache(maxsize=2)
def _load_codeswitch_lexicon() -> dict[str, str]:
    """Load the bundled English code-switching lexicon."""
    if not _LEXICON_PATH.exists():
        try:
            from africa_g2p.codeswitch import load_codeswitch_dict
            return load_codeswitch_dict()
        except ImportError:
            return {}
    lexicon = {}
    with gzip.open(_LEXICON_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2:
                lexicon[p[0].lower()] = p[1]
    return lexicon


def _convert_codeswitch(text: str, lang: str, lexicon: dict[str, str]) -> str:
    """Greedy longest-match code-switching: prioritises longer phrases (up to 5 words)
    before shorter individual words."""
    words = text.split()
    out = []
    i = 0
    n = len(words)
    while i < n:
        matched = False
        # Try window sizes from 5 words down to 1 word
        for k in (5, 4, 3, 2, 1):
            if i + k <= n:
                window = words[i:i + k]
                lead, first_core, _ = _PUNCT.match(window[0]).groups()
                _, last_core, trail = _PUNCT.match(window[-1]).groups()
                if k == 1:
                    phrase = first_core.lower()
                else:
                    middle = [_PUNCT.match(w).groups()[1].lower() for w in window[1:-1]]
                    phrase = " ".join([first_core.lower()] + middle + [last_core.lower()])

                if phrase in lexicon:
                    out.append(lead + lexicon[phrase] + trail)
                    i += k
                    matched = True
                    break
        if not matched:
            lead, core, trail = _PUNCT.match(words[i]).groups()
            if core:
                ipa = g2p(core, lang, output="ipa")
                out.append(lead + ipa + trail)
            else:
                out.append(words[i])
            i += 1
    return " ".join(out)


def to_ipa(text: str, lang: str, codeswitch: bool = True) -> str:
    """Convert `text` in language `lang` (ISO 639-3 code, e.g. "twi", "eng") to model IPA.

    `codeswitch`: if True (default), embedded English loanwords, nouns and named entities
    are detected using longest-match phrase lookup (up to 5 words before 1 word) and
    phonemized using English IPA (espeak-ng) while surrounding words use native African G2P.
    """
    clean_text = text.translate(_LOOKALIKES)
    if lang != "eng":
        if codeswitch:
            lexicon = _load_codeswitch_lexicon()
            if lexicon:
                return _convert_codeswitch(clean_text, lang, lexicon)
        return g2p(clean_text, lang, output="ipa")
    words = []
    for w in text.split():
        lead, core, trail = _PUNCT.match(w).groups()
        ipa = g2p(core, "eng", output="ipa").replace(" ", "") if core else ""
        words.append(lead + ipa + trail)
    return " ".join(words)
