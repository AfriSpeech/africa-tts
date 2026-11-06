"""Check a front-end build against the IPA the model was trained on.

    python check_parity.py ATT_IPA_BINARY G2P_DIR [ESPEAK_DIR]

ATT_IPA_BINARY: the att_ipa tool built from this folder. G2P_DIR: the <lang>.afg2p tables
(hf://AfriSpeech/africa-tts/g2p/). ESPEAK_DIR: folder containing espeak-ng-data (omit for the
system default). Exits non-zero unless every row matches.

espeak-ng must be version 1.51: 1.52 merged the vowels of "four" (foːɹ) and "for" (fɔːɹ), and
about 2% of English words come out differently from what the model was trained on.
"""

import subprocess
import sys
from pathlib import Path

rows = [line.rstrip("\n").split("\t") for line in
        (Path(__file__).parent / "parity_set.tsv").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")]
cmd = [sys.argv[1], "-g", sys.argv[2], "-t"] + (["-e", sys.argv[3]] if len(sys.argv) > 3 else [])
out = subprocess.run(cmd, input="".join(f"{l}\t{t}\n" for l, t, _ in rows),
                     capture_output=True, text=True).stdout.split("\n")
bad = [(l, t, e, g) for (l, t, e), g in zip(rows, out) if g != e]
for lang in ("eng", None):
    sel = [r for r in rows if (r[0] == "eng") == (lang == "eng")]
    nbad = sum(1 for b in bad if (b[0] == "eng") == (lang == "eng"))
    print(f"{'English words' if lang else 'African sentences'}: {len(sel) - nbad}/{len(sel)} identical")
for l, t, e, g in bad[:10]:
    print(f"  {l} {t!r}\n    expected {e!r}\n    got      {g!r}")
sys.exit(1 if bad else 0)
