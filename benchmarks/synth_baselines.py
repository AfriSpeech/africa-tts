"""Synthesise the benchmark test set with baseline systems, on the languages each supports.

  mms        facebook/mms-tts-<iso> (VITS, one voice per language; uroman input where the model
             expects it)
  omnivoice  k2-fsa/OmniVoice, zero-shot, cloning the SAME default voice clip (and its
             transcript) that Africa TTS uses for the language, with the language id set

Usage: python synth_baselines.py BENCH_DIR VOICES_DIR --system mms|omnivoice [--device cuda]
  VOICES_DIR: Africa TTS default voices (<lang>.wav + voices.json with each clip's transcript).
Writes BENCH_DIR/synth/<system>/<lang>/<id>.wav
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MMS = {"amh": "amh", "bem": "bem", "dyu": "dyu", "eng": "eng", "ewe": "ewe", "fon": "fon",
       "hau": "hau", "kab": "kab", "kbp": "kbp", "kik": "kik", "kin": "kin", "lug": "lug",
       "mfe": "mfe", "mlg": "mlg", "mos": "mos", "nya": "nya", "run": "run", "sag": "sag",
       "seh": "seh", "sna": "sna", "swh": "swh", "tir": "tir", "tso": "tso", "vmw": "vmw",
       "yor": "yor", "twi": "aka"}
OMNIVOICE = {"amh": "am", "cjk": "cjk", "dyu": "dyu", "eng": "en", "hau": "ha", "ibo": "ig",
             "kab": "kab", "kam": "kam", "kea": "kea", "kin": "rw", "lin": "ln", "lug": "lg",
             "mlg": "plt", "nso": "nso", "nya": "ny", "sna": "sn", "swh": "sw", "tir": "ti",
             "twi": "tw", "umb": "umb", "wol": "wo", "xho": "xh", "yor": "yo", "zul": "zu",
             "fat": "fat", "gur": "gur", "her": "hz", "iso": "iso", "mer": "mer", "ndo": "ng",
             "xpe": "xpe"}


def run_mms(by_lang, out_root: Path, device: str) -> None:
    import uroman
    from transformers import AutoTokenizer, VitsModel

    ur = uroman.Uroman()
    for lang, rows in by_lang.items():
        if lang not in MMS:
            continue
        name = f"facebook/mms-tts-{MMS[lang]}"
        tok = AutoTokenizer.from_pretrained(name)
        model = VitsModel.from_pretrained(name).to(device).eval()
        for r in rows:
            out = out_root / lang / f"{r['id']}.wav"
            if out.exists():
                continue
            text = r["text"]
            if getattr(tok, "is_uroman", False):
                text = ur.romanize_string(text, lcode=MMS[lang])
            inputs = tok(text, return_tensors="pt").to(device)
            if inputs["input_ids"].shape[1] == 0:
                continue
            torch.manual_seed(0)
            with torch.no_grad():
                wav = model(**inputs).waveform[0].float().cpu().numpy()
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, wav, model.config.sampling_rate)
        print(f"mms {lang}: {len(rows)}", flush=True)
        del model


def run_omnivoice(by_lang, out_root: Path, voices: Path, device: str) -> None:
    from omnivoice import OmniVoice

    meta = json.loads((voices / "voices.json").read_text())
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=torch.float16)
    for lang, rows in by_lang.items():
        if lang not in OMNIVOICE:
            continue
        prompt = model.create_voice_clone_prompt(ref_audio=str(voices / f"{lang}.wav"),
                                                 ref_text=meta[lang]["text"])
        for r in rows:
            out = out_root / lang / f"{r['id']}.wav"
            if out.exists():
                continue
            torch.manual_seed(0)
            audio = model.generate(text=r["text"], language=OMNIVOICE[lang], voice_clone_prompt=prompt)
            wav = audio[0] if isinstance(audio, list) else audio
            wav = wav.float().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav)
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, wav.squeeze(), model.sampling_rate)
        print(f"omnivoice {lang}: {len(rows)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path)
    ap.add_argument("voices", type=Path)
    ap.add_argument("--system", required=True, choices=["mms", "omnivoice"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    by_lang = defaultdict(list)
    for line in open(a.bench / "testset.jsonl"):
        r = json.loads(line)
        by_lang[r["lang"]].append(r)
    out_root = a.bench / "synth" / a.system
    if a.system == "mms":
        run_mms(by_lang, out_root, a.device)
    else:
        run_omnivoice(by_lang, out_root, a.voices, a.device)


if __name__ == "__main__":
    main()
