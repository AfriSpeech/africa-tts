"""Out-of-domain synthesis demo: FLORES+ devtest sentences (news/wiki, outside the training
domain) for every training language FLORES+ covers; held-out validation sentences for the rest.

The same FLORES sentence ids are used in every language (parallel, with the English as a
gloss), chosen digit-free in all languages and of moderate length. Each language is read in
the voice of one of its validation clips (first <= 5 s). One checkpoint, EMA weights,
cfg 2.0, temperature 0.3. Text -> IPA exactly as in training (africa-g2p C tables per word;
English via espeak-ng + africa-g2p normalisation).

Usage: python synth_ood.py CONFIG CKPT OUT [--n 3]
"""

import argparse
import json
import random
import re
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from paths import AFRICA_G2P, DATA, RECIPE, REPO  # noqa: E402
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RECIPE / "release"))
sys.path.insert(0, str(RECIPE / "data"))
from pocket_tts.modules.stateful_module import init_states  # noqa: E402
from synth_demo import VALID, window  # noqa: E402
from to_ipa import TABLES, Afrig2p, English  # noqa: E402
from training.args import load_args  # noqa: E402
from training.modules.builders import build_models  # noqa: E402

REPO = "openlanguagedata/flores_plus"
PREFER = {"twi": "twi_Latn_asan1239"}  # Asante Twi, like our Twi data
LANGS_JW = ["fat", "gaa"]  # not in FLORES+: keep held-out JW sentences
FIRST = ["twi", "fat", "gaa", "ewe", "hau", "yor", "ibo", "swh", "zul", "xho", "amh", "kin",
         "sna", "lin", "fon", "eng"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    (a.out / "audio").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    valid = defaultdict(list)
    names = {}
    for line in open(VALID, encoding="utf-8"):
        r = json.loads(line)
        names[r["g2p"]] = r["language"]
        if 3.0 <= r["duration"] <= 11.0:
            valid[r["g2p"]].append(r)

    files = {}
    for f in sorted(list_repo_files(REPO, repo_type="dataset")):
        if f.startswith("devtest/"):
            files.setdefault(f.split("/")[1].split("_")[0], f)
    files.update({g: f"devtest/{n}.jsonl" for g, n in PREFER.items()})
    langs = [g for g in valid if g in files and g != "eng"] + ["eng"]
    langs = [g for g in FIRST if g in langs or g in LANGS_JW] + sorted(set(langs) - set(FIRST))
    flores = {}
    for g in langs:
        if g in files:
            p = hf_hub_download(REPO, files[g], repo_type="dataset")
            flores[g] = {r["id"]: r["text"] for r in map(json.loads, open(p, encoding="utf-8"))}
    ids = [i for i, t in flores["eng"].items() if 60 <= len(t) <= 130
           and not any(re.search(r"\d", flores[g].get(i, "0")) for g in flores)]
    ids = rng.sample(ids, a.n)
    print("sentence ids:", ids)

    eng = English()
    eng.batch([w for i in ids for w in flores["eng"][i].split()])
    items = []
    for g in langs:
        voice = rng.choice(valid[g])
        vname = f"{g}_voice.wav"
        sf.write(a.out / "audio" / vname, window(voice, 5.0), 24000)
        if g in flores:
            sents = [(flores[g][i], None, flores["eng"][i]) for i in ids]
        else:  # held-out validation sentences, with the real recording
            rows = rng.sample([r for r in valid[g] if r is not voice], a.n)
            sents = [(r["transcript_orig"], r, None) for r in rows]
        table = None if g == "eng" else Afrig2p(TABLES / f"{g}.afg2p")
        for k, (text, row, gloss) in enumerate(sents):
            ipa = row["transcript"] if row else " ".join(
                eng.word(w) if g == "eng" else table.ipa(w) for w in text.split())
            real = None
            if row:
                real = f"{g}_{k}_real.wav"
                sf.write(a.out / "audio" / real, window(row), 24000)
            items.append({"g2p": g, "language": names[g], "k": k, "text": text, "ipa": ipa,
                          "gloss": gloss, "source": "FLORES+" if row is None else "held out",
                          "real": real, "voice": vname, "voice_row": voice, "synth": {}})

    args = load_args(a.config)
    device = torch.device("cuda")
    model, mimi, _ = build_models(args)
    payload = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    state = dict(payload["model"])
    if payload.get("ema"):
        state.update(payload["ema"])
    model.load_state_dict(state)
    model, mimi = model.to(device).eval(), mimi.to(device).eval()
    tokenize = model.flow_lm.conditioner.tokenizer.encode
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    voices = {}
    with torch.no_grad():
        for it in items:
            if it["voice"] not in voices:
                w = torch.from_numpy(window(it["voice_row"], 5.0))[None, None].to(device)
                voices[it["voice"]] = mimi.encode_to_latent(w)[0].float()
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
                fname = f"{it['g2p']}_{it['k']}_synth.wav"
                sf.write(a.out / "audio" / fname, audio.float().cpu().numpy(), mimi.sample_rate)
                it["synth"]["100k"] = fname
            print(f"{b + len(batch)}/{len(items)}", flush=True)
    for it in items:
        it.pop("voice_row")
    (a.out / "meta.json").write_text(json.dumps({"checkpoints": ["100k"], "items": items},
                                                ensure_ascii=False, indent=1))
    print(f"{len(items)} sentences, {len(langs)} languages -> {a.out}")


if __name__ == "__main__":
    main()
