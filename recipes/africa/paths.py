"""Where the African recipe looks for things. Override with environment variables.

  AFRICA_TTS_DATA   data root (manifests, audio, aligned/IPA files)      default: <repo>/data
  AFRICA_G2P_DIR    checkout of AfriSpeech/africa-g2p at commit 76b53a8,
                    with native/ built and its tables exported           default: <repo>/third_party/africa-g2p
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AFRICA_TTS_DATA", REPO / "data"))
AFRICA_G2P = Path(os.environ.get("AFRICA_G2P_DIR", REPO / "third_party" / "africa-g2p"))
RECIPE = REPO / "recipes" / "africa"
