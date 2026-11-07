"""UDHR samples: Articles 1, 3 and 5 of the Universal Declaration of Human Rights (Unicode
UDHR collection, complete translations only) in every training language that has one.

Each language is read in the voice of one of its clips (validation if any, else training;
first <= 5 s). Text -> IPA exactly as in training. EMA weights.

Usage: python synth_udhr.py CONFIG CKPT OUT [--label teacher] [--cfg 2.0]
  For the distilled student pass its own (non-distill) config and --cfg 1.0.
"""

import argparse
import json
import random
import re
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from paths import AFRICA_G2P, DATA, RECIPE, REPO  # noqa: E402
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RECIPE / "release"))
sys.path.insert(0, str(RECIPE / "data"))
from pocket_tts.modules.stateful_module import init_states  # noqa: E402
from synth_demo import VALID, window  # noqa: E402
from to_ipa import TABLES, Afrig2p, English  # noqa: E402
from training.args import load_args  # noqa: E402
from training.modules.builders import build_models  # noqa: E402

UDHR = (DATA / "udhr/udhr-main/data/udhr")
TRAIN = str((DATA / "africa-tts/ipa/train_all.jsonl"))
NS = "{http://efele.net/udhr}"
ARTICLES = ["1", "3", "5"]
PREFER = {"twi": "aka_asante", "eng": "eng"}
CODE_ALIAS = {"mlg": ["plt", "mlg"], "swh": ["swh", "swa"]}
FIRST = ["twi", "fat", "gaa", "ewe", "dag", "hau", "yor", "ibo", "swh", "zul", "xho", "amh",
         "kin", "sna", "lin", "fon", "eng"]


def udhr_index() -> dict[str, list[str]]:
    by: dict[str, list[str]] = defaultdict(list)
    for e in ET.parse(UDHR / "index.xml").getroot().iter("udhr"):
        if int(e.get("stage", "0")) >= 4:
            by[e.get("iso639-3")].append(e.get("f"))
    return by


def articles(key: str) -> list[str] | None:
    root = ET.parse(UDHR / f"udhr_{key}.xml").getroot()
    out = []
    for art in root.iter(f"{NS}article"):
        if art.get("number") in ARTICLES:
            text = " ".join(" ".join(p.itertext()) for p in art.iter(f"{NS}para"))
            text = re.sub(r"\s+", " ", text).strip()
            if not text or re.search(r"\d", text):
                return None
            out.append(text)
    return out if len(out) == len(ARTICLES) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("out", type=Path)
    ap.add_argument("--label", default="teacher")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--reuse-meta", action="store_true",
                    help="keep texts/voices from OUT/meta.json, only add this label's audio")
    a = ap.parse_args()
    (a.out / "audio").mkdir(parents=True, exist_ok=True)
    meta_path = a.out / "meta.json"

    if a.reuse_meta:
        meta = json.loads(meta_path.read_text())
        items = meta["items"]
        voice_rows = meta.pop("_voice_rows")
    else:
        rng = random.Random(a.seed)
        clips, names = defaultdict(list), {}
        for src in (VALID, TRAIN):
            for line in open(src, encoding="utf-8"):
                r = json.loads(line)
                names.setdefault(r["g2p"], r["language"])
                if 4.0 <= r["duration"] <= 15.0 and len(clips[r["g2p"]]) < 400:
                    clips[r["g2p"]].append(r)
        index = udhr_index()
        langs = sorted(clips, key=lambda g: (g not in FIRST, FIRST.index(g) if g in FIRST else 0,
                                             names[g]))
        eng = English()
        items, voice_rows, skipped = [], {}, []
        for g in langs:
            # The collection files several translations under one code (e.g. swh: Chimwiini,
            # Kibajuni and standard Swahili); prefer the one whose file is named after the code.
            keys = [PREFER[g]] if g in PREFER else [k for c in CODE_ALIAS.get(g, [g])
                                                    for k in sorted(index.get(c, []),
                                                                    key=lambda f, c=c: f != c)]
            texts = next((t for k in keys if (t := articles(k))), None)
            if texts is None:
                skipped.append(g)
                continue
            voice = rng.choice(clips[g])
            vname = f"{g}_voice.wav"
            sf.write(a.out / "audio" / vname, window(voice, 5.0), 24000)
            voice_rows[vname] = voice
            if g == "eng":
                eng.batch([w for t in texts for w in t.split()])
            table = None if g == "eng" else Afrig2p(TABLES / f"{g}.afg2p")
            for art, text in zip(ARTICLES, texts):
                ipa = " ".join(eng.word(w) if g == "eng" else table.ipa(w) for w in text.split())
                items.append({"g2p": g, "language": names[g], "article": art, "text": text,
                              "ipa": ipa, "voice": vname, "synth": {}})
        meta = {"labels": [], "items": items}
        print(f"{len({i['g2p'] for i in items})} languages with UDHR; no UDHR text: {skipped}")

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
        for v, row in voice_rows.items():
            w = torch.from_numpy(window(row, 5.0))[None, None].to(device)
            voices[v] = mimi.encode_to_latent(w)[0].float()
        for b in range(0, len(items), 16):
            batch = items[b:b + 16]
            toks = [torch.tensor(tokenize(it["ipa"]), dtype=torch.long) for it in batch]
            outs = model.generate(toks, [voices[it["voice"]] for it in batch], temp=0.3,
                                  cfg_coef=a.cfg)
            for it, lat in zip(batch, outs):
                if lat.shape[0] < 8:
                    continue
                st = init_states(mimi, 1, (lat.shape[0] + 8) * ratio)
                audio = mimi.decode_from_latent(lat[None].to(device), st)[0, 0]
                fname = f"{it['g2p']}_art{it['article']}_{a.label}.wav"
                sf.write(a.out / "audio" / fname, audio.float().cpu().numpy(), mimi.sample_rate)
                it["synth"][a.label] = fname
            print(f"{b + len(batch)}/{len(items)}", flush=True)
    if a.label not in meta["labels"]:
        meta["labels"].append(a.label)
    meta["_voice_rows"] = voice_rows
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"{len(items)} sentences -> {a.out}")


if __name__ == "__main__":
    main()
