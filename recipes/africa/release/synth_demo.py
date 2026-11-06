"""Synthesize held-out validation sentences in many languages with several teacher checkpoints.

For each language: N sentences from the validation set (never trained on), each read in the
voice of a DIFFERENT validation clip of the same language (first <= 5 s as the voice prompt).
Checkpoints are loaded with their EMA weights; generation is the trainer's own (cfg 2.0,
temperature 0.3). Writes OUT/audio/*.wav and OUT/meta.json for the demo page.

Usage: python synth_demo.py CONFIG OUT --ckpt NAME=PATH ... [--per-lang 2]
"""

import argparse
import json
import random
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from paths import AFRICA_G2P, DATA, RECIPE, REPO  # noqa: E402
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(REPO))
from pocket_tts.modules.stateful_module import init_states  # noqa: E402
from training.args import load_args  # noqa: E402
from training.modules.builders import build_models  # noqa: E402

LANGS = ["twi", "fat", "ewe", "gaa", "dag", "hau", "yor", "ibo", "swh", "zul", "xho", "amh",
         "kin", "sna", "lin", "fon", "eng"]
VALID = str((DATA / "africa-tts/ipa/valid_all.jsonl"))


def window(r: dict, max_sec: float | None = None) -> np.ndarray:
    sr = sf.info(r["path"]).samplerate
    dur = r["duration"] if max_sec is None else min(r["duration"], max_sec)
    w, _ = sf.read(r["path"], start=int(r["start"] * sr), frames=int(dur * sr), dtype="float32",
                   always_2d=True)
    w = w.mean(1)
    if sr != 24000:
        from scipy.signal import resample_poly
        g = np.gcd(sr, 24000)
        w = resample_poly(w, 24000 // g, sr // g).astype(np.float32)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out", type=Path)
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--per-lang", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    (a.out / "audio").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    by = defaultdict(list)
    for line in open(VALID, encoding="utf-8"):
        r = json.loads(line)
        if 3.0 <= r["duration"] <= 11.0:
            by[r["g2p"]].append(r)
    items = []
    for g in LANGS:
        rows = by.get(g, [])
        if len(rows) < a.per_lang + 1:
            print(f"skip {g}: {len(rows)} validation clips")
            continue
        picks = rng.sample(rows, a.per_lang + 1)
        voice, sents = picks[0], picks[1:]
        vname = f"{g}_voice.wav"
        sf.write(a.out / "audio" / vname, window(voice, 5.0), 24000)
        for k, r in enumerate(sents):
            rname = f"{g}_{k}_real.wav"
            sf.write(a.out / "audio" / rname, window(r), 24000)
            items.append({"g2p": g, "language": r["language"], "k": k,
                          "text": r["transcript_orig"], "ipa": r["transcript"], "real": rname,
                          "voice": vname, "voice_row": voice, "synth": {}})

    args = load_args(a.config)
    device = torch.device("cuda")
    model, mimi, _ = build_models(args)
    model, mimi = model.to(device).eval(), mimi.to(device).eval()
    tokenize = model.flow_lm.conditioner.tokenizer.encode
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    voices = {}
    with torch.no_grad():
        for it in items:
            if it["voice"] not in voices:
                w = torch.from_numpy(window(it["voice_row"], 5.0))[None, None].to(device)
                voices[it["voice"]] = mimi.encode_to_latent(w)[0].float()
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = dict(payload["model"])
        if payload.get("ema"):
            state.update(payload["ema"])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            for b in range(0, len(items), 16):
                batch = items[b:b + 16]
                toks = [torch.tensor(tokenize(it["ipa"]), dtype=torch.long) for it in batch]
                outs = model.generate(toks, [voices[it["voice"]] for it in batch], temp=0.3,
                                      cfg_coef=2.0)
                for it, lat in zip(batch, outs):
                    if lat.shape[0] < 8:
                        continue
                    st = init_states(mimi, 1, (lat.shape[0] + 8) * ratio)
                    audio = mimi.decode_from_latent(lat[None].to(device), st)[0, 0]
                    fname = f"{it['g2p']}_{it['k']}_{name}.wav"
                    sf.write(a.out / "audio" / fname, audio.float().cpu().numpy(), mimi.sample_rate)
                    it["synth"][name] = fname
        print(f"{name}: done", flush=True)
    for it in items:
        it.pop("voice_row")
    (a.out / "meta.json").write_text(json.dumps({"checkpoints": [s.split("=")[0] for s in a.ckpt],
                                                 "items": items}, ensure_ascii=False, indent=1))
    print(f"{len(items)} sentences, {len({i['g2p'] for i in items})} languages -> {a.out}")


if __name__ == "__main__":
    main()
