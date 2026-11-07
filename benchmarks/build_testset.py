"""Build the Africa TTS benchmark test set: per language, N sentences to synthesise, with real
recordings where the source has them (the ASR baseline).

Sources, in order of preference for each of our languages:
  1. facebook/omnilingual-asr-corpus  (real speech; test split, else dev, else train)
  2. google/fleurs                    (real read speech, test split; FLORES sentences)
  3. google/WaxalNLP                  (real speech, test split: ASR set, else TTS set)
  4. openlanguagedata/flores_plus     (devtest text only: no real-audio baseline)
Sentences with digits are dropped (the models never saw numerals; spelling numbers out is
language-specific), as are corpus segments with transcription tags or longer than 25 s.

Usage: python build_testset.py OUT_DIR [--n 50] [--langs abr,guz,...]
(--langs: other languages, e.g. experimental ones never trained on; default: all trained ones)
Writes OUT_DIR/testset.jsonl and OUT_DIR/real/<lang>/<id>.wav (16 kHz mono).
"""

import argparse
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

PKG = Path(__file__).resolve().parents[1] / "africa_tts"
OMNI = "facebook/omnilingual-asr-corpus"
FLEURS = "google/fleurs"
FLORES = "openlanguagedata/flores_plus"
WAXAL = "google/WaxalNLP"
WAXAL_CODE = {"twi": "aka", "swh": "swa"}  # our code -> WAXAL folder

# our code -> (iso in the source datasets, script)
ISO = {"mlg": "plt", "daf": "dnj"}
FLEURS_CFG = {"amh": "am_et", "hau": "ha_ng", "ibo": "ig_ng", "kam": "kam_ke", "kea": "kea_cv",
              "lug": "lg_ug", "lin": "ln_cd", "nso": "nso_za", "nya": "ny_mw", "sna": "sn_zw",
              "swh": "sw_ke", "umb": "umb_ao", "wol": "wo_sn", "xho": "xh_za", "yor": "yo_ng",
              "zul": "zu_za", "eng": "en_us", "afr": "af_za"}
FLORES_FILE = {"twi": "twi_Latn_asan1239", "tso": "tso_Latn_MZ", "seh": "seh_Latn_MZ",
               "nya": "nya_Latn", "knc": "knc_Latn"}
DIGIT = re.compile(r"\d")


def ok_text(t: str) -> bool:
    return 20 <= len(t) <= 220 and not DIGIT.search(t)


def to16k(arr: np.ndarray, sr: int) -> np.ndarray:
    if arr.ndim > 1:
        arr = arr.mean(1)
    if sr == 16000:
        return arr.astype(np.float32)
    n = int(round(len(arr) * 16000 / sr))
    return np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr).astype(np.float32)


def omni_rows(code: str, files: list[str], n: int, rng: random.Random, fill: bool = False):
    """Rows from the first split (test, dev, train) that has any; with fill, top up from the
    following splits until n (the source is then reported as e.g. "test+dev")."""
    iso = ISO.get(code, code)
    mine = [f for f in files if f"/{iso}_" in f and f.endswith(".parquet")]
    if fill:
        got, used = [], []
        for split in ("test", "dev", "train"):
            if len(got) >= n:
                break
            sp, rows = omni_rows(code, [f for f in mine if f.split("/")[-1].startswith(split + "-")],
                                 n - len(got), rng)
            if rows:
                got += rows
                used.append(sp)
        return "+".join(used) or None, got
    for split in ("test", "dev", "train"):
        sf_ = sorted(f for f in mine if f.split("/")[-1].startswith(split + "-"))
        if not sf_:
            continue
        rows = []
        for f in sf_:
            df = pd.read_parquet(hf_hub_download(OMNI, f, repo_type="dataset"))
            for r in df.itertuples():
                t = " ".join(str(r.raw_text).split())
                if r.duration <= 25 and ok_text(t) and not re.search(r"[<>\[\]]", t):
                    rows.append((f"{r.speaker_id}_{r.prompt_id}_{r.segment_id}", t, r.audio, r.iso_15924))
            if len(rows) >= 4 * n:
                break
        if rows:
            rng.shuffle(rows)
            return split, rows[:n]
    return None, []


def fleurs_rows(cfg: str, n: int, rng: random.Random):
    df = pd.read_parquet(hf_hub_download(FLEURS, f"parquet-data/{cfg}/test-00000-of-00001.parquet",
                                         repo_type="dataset"))
    seen, rows = set(), []
    for r in df.itertuples():
        t = " ".join(str(r.raw_transcription).split())
        if r.id in seen or not ok_text(t):  # FLEURS has several readings per sentence: keep one
            continue
        seen.add(r.id)
        rows.append((f"fleurs{r.id}", t, r.audio))
    rng.shuffle(rows)
    return rows[:n]


def waxal_rows(code: str, files: list[str], n: int, rng: random.Random):
    """Test-split rows of WAXAL's ASR set (natural speech), else its TTS set (read speech)."""
    w = WAXAL_CODE.get(code, code)
    for kind, col in (("ASR", "transcription"), ("TTS", "text")):
        test = sorted(f for f in files if f.startswith(f"data/{kind}/{w}/")
                      and f.split("/")[-1].replace(f"{w}-", "").startswith("test"))
        rows = []
        for f in test:
            df = pd.read_parquet(hf_hub_download(WAXAL, f, repo_type="dataset"))
            for r in df.itertuples():
                t = " ".join(str(getattr(r, col)).split())
                if ok_text(t) and not re.search(r"[<>\[\]]", t):
                    rows.append((f"waxal_{r.id}", t, r.audio))
            if len(rows) >= 4 * n:
                break
        if rows:
            rng.shuffle(rows)
            return f"waxal-{kind.lower()}/test", rows[:n]
    return None, []


def flores_rows(fname: str, n: int, rng: random.Random):
    p = hf_hub_download(FLORES, f"devtest/{fname}.jsonl", repo_type="dataset")
    rows = [(f"flores{r['id']}", " ".join(r["text"].split())) for r in map(json.loads, open(p))]
    rows = [r for r in rows if ok_text(r[1])]
    rng.shuffle(rows)
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--fill-splits", action="store_true",
                    help="top up a language from the corpus's dev/train splits when test has < N")
    ap.add_argument("--langs", default=None, help="comma-separated codes (default: languages.json)")
    a = ap.parse_args()
    ours = (a.langs.split(",") if a.langs else json.loads((PKG / "languages.json").read_text()))
    omni_files = list_repo_files(OMNI, repo_type="dataset")
    waxal_files = list_repo_files(WAXAL, repo_type="dataset")
    flores_files = {f.split("/")[1][:-6] for f in list_repo_files(FLORES, repo_type="dataset")
                    if f.startswith("devtest/")}
    out = []
    for code in ours:
        rng = random.Random(f"africa-tts-bench-{code}")
        iso = ISO.get(code, code)
        split, rows = omni_rows(code, omni_files, a.n, rng, a.fill_splits)
        if rows:
            src = f"omnilingual-asr-corpus/{split}"
            for rid, t, audio, script in rows:
                arr, sr = sf.read(io.BytesIO(audio["bytes"]))
                wav = a.out / "real" / code / f"{rid}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav, to16k(arr, sr), 16000)
                out.append({"lang": code, "id": rid, "text": t, "source": src,
                            "real": str(wav.relative_to(a.out)), "script": script})
        elif code in FLEURS_CFG:
            for rid, t, audio in fleurs_rows(FLEURS_CFG[code], a.n, rng):
                arr, sr = sf.read(io.BytesIO(audio["bytes"]))
                wav = a.out / "real" / code / f"{rid}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav, to16k(arr, sr), 16000)
                out.append({"lang": code, "id": rid, "text": t, "source": "fleurs/test",
                            "real": str(wav.relative_to(a.out)), "script": None})
        elif (wx := waxal_rows(code, waxal_files, a.n, rng))[1]:
            for rid, t, audio in wx[1]:
                arr, sr = sf.read(io.BytesIO(audio["bytes"]))
                wav = a.out / "real" / code / f"{rid}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav, to16k(arr, sr), 16000)
                out.append({"lang": code, "id": rid, "text": t, "source": wx[0],
                            "real": str(wav.relative_to(a.out)), "script": None})
        else:
            fname = FLORES_FILE.get(code) or next(
                (f for f in sorted(flores_files) if f.startswith(iso + "_")), None)
            if not fname or fname not in flores_files:
                continue
            for rid, t in flores_rows(fname, a.n, rng):
                out.append({"lang": code, "id": rid, "text": t, "source": "flores_plus/devtest",
                            "real": None, "script": fname.split("_")[1]})
        print(code, sum(1 for r in out if r["lang"] == code), out[-1]["source"] if out else "", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "testset.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    langs = sorted({r["lang"] for r in out})
    print(f"{len(out)} sentences, {len(langs)} languages")


if __name__ == "__main__":
    main()
