"""Replace default voices with female ones from AfriSpeech/africa-female-speech-v2-best60, for the
trained languages that dataset covers.

best60 holds the 60 best clips per language (24 kHz, Demucs-cleaned, UTMOS-scored). Per language
we take the highest-UTMOS clip of 3.5-6.5 s with no digit in its text (the prompts the model was
trained with were at most 5-6 s), trim leading/trailing silence to 0.15 s, and write
VOICES/<code>.wav, updating VOICES/voices.json. The replaced voice is kept as VOICES/jw/<code>.wav.

Usage: python select_female_voices.py VOICES_DIR
"""

import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files
from scipy.signal import resample_poly

REPO = "AfriSpeech/africa-female-speech-v2-best60"
CODE_MAP = {"swa": "swh"}  # dataset code -> model code
MIN_S, MAX_S, PAD = 3.5, 6.5, 0.15


def trim(a: np.ndarray, sr: int) -> np.ndarray:
    frame = int(0.02 * sr)
    n = len(a) // frame
    if n == 0:
        return a
    rms = np.sqrt((a[: n * frame].reshape(n, frame) ** 2).mean(1))
    voiced = np.where(rms > max(rms.max() * 0.05, 1e-4))[0]
    if len(voiced) == 0:
        return a
    s = max(0, voiced[0] * frame - int(PAD * sr))
    e = min(len(a), (voiced[-1] + 1) * frame + int(PAD * sr))
    return a[s:e]


def main() -> None:
    voices = Path(sys.argv[1])
    meta_path = voices / "voices.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    (voices / "jw").mkdir(exist_ok=True)
    configs = sorted({f.split("/")[0] for f in list_repo_files(REPO, repo_type="dataset") if "/" in f})
    done = []
    for cfg in configs:
        code = CODE_MAP.get(cfg.rsplit("_", 1)[-1], cfg.rsplit("_", 1)[-1])
        if code not in meta:
            print(f"skip {cfg}: not a trained language")
            continue
        table = pq.read_table(hf_hub_download(REPO, f"{cfg}/train-00000-of-00001.parquet",
                                              repo_type="dataset")).to_pylist()
        cands = sorted((r for r in table if MIN_S <= r["duration"] <= MAX_S
                        and not re.search(r"\d", r["text"])), key=lambda r: -r["utmos"])
        if not cands:
            print(f"skip {cfg}: no clip of {MIN_S}-{MAX_S} s")
            continue
        r = cands[0]
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        a = a.mean(1)
        if sr != 24000:
            g = np.gcd(sr, 24000)
            a = resample_poly(a, 24000 // g, sr // g).astype(np.float32)
        a = trim(a, 24000)
        if (voices / f"{code}.wav").exists() and not (voices / "jw" / f"{code}.wav").exists():
            shutil.copy(voices / f"{code}.wav", voices / "jw" / f"{code}.wav")
        sf.write(voices / f"{code}.wav", a, 24000, subtype="PCM_16")
        meta[code] = {"language": meta[code]["language"], "source": "africa-female-speech-v2-best60",
                      "subset": cfg, "clip": r["id"], "duration": round(len(a) / 24000, 2),
                      "text": r["text"], "utmos": round(r["utmos"], 3), "speaker": "female",
                      "previous": meta[code]}
        done.append(f"{code} ({len(a) / 24000:.1f} s, UTMOS {r['utmos']:.2f})")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(done)} voices replaced: " + ", ".join(done))


if __name__ == "__main__":
    main()
