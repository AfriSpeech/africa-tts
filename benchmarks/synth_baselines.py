"""Synthesise the benchmark test set with baseline systems, on the languages each supports.

  mms        facebook/mms-tts-<iso> (VITS, one voice per language; uroman input where the model
             expects it)
  omnivoice  k2-fsa/OmniVoice, zero-shot, cloning the SAME default voice clip (and its
             transcript) that Africa TTS uses for the language, with the language id set
  orpheus    Sunbird/orpheus-3b-tts-multilingual (Llama-3 3B + SNAC), one fixed speaker per
             language from its model card

Usage: python synth_baselines.py BENCH_DIR VOICES_DIR --system mms|omnivoice|orpheus [--device cuda]
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
       "yor": "yor", "twi": "aka", "dga": "dga",
       # experimental (unseen by Africa TTS) languages of the unseen-language benchmark
       "ach": "ach", "bam": "bam"}
OMNIVOICE = {"amh": "am", "cjk": "cjk", "dyu": "dyu", "eng": "en", "hau": "ha", "ibo": "ig",
             "kab": "kab", "kam": "kam", "kea": "kea", "kin": "rw", "lin": "ln", "lug": "lg",
             "mlg": "plt", "nso": "nso", "nya": "ny", "sna": "sn", "swh": "sw", "tir": "ti",
             "twi": "tw", "umb": "umb", "wol": "wo", "xho": "xh", "yor": "yo", "zul": "zu",
             "fat": "fat", "gur": "gur", "her": "hz", "iso": "iso", "mer": "mer", "ndo": "ng",
             "xpe": "xpe",
             # experimental (unseen by Africa TTS) languages of the unseen-language benchmark
             "afr": "af", **{c: c for c in ("abr", "cgg", "dag", "ebu", "fuc", "fuf", "gol", "guz",
                                            "idu", "ijn", "koo", "lag", "maf", "snk", "vai")}}


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


def run_omnivoice(by_lang, out_root: Path, voices: Path, device: str,
                  voice_texts: Path | None = None) -> None:
    from omnivoice import OmniVoice

    meta = json.loads((voices / "voices.json").read_text())
    if voice_texts:  # prompt transcripts from transcribe_voices.py (JW clips' stored text is partial)
        for v, t in json.loads(voice_texts.read_text()).items():
            meta[v]["text"] = t["text"]
    exp = json.loads((Path(__file__).resolve().parents[1] / "africa_tts" /
                      "languages_experimental.json").read_text())
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=torch.float16)
    for lang, rows in by_lang.items():
        if lang not in OMNIVOICE:
            continue
        # the same clip Africa TTS uses: for an experimental language, its closest trained one's
        v = exp[lang]["nearest"] if lang in exp else lang
        prompt = model.create_voice_clone_prompt(ref_audio=str(voices / f"{v}.wav"),
                                                 ref_text=meta[v]["text"])
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


# Sunbird/orpheus-3b-tts-multilingual: no voice cloning; one fixed speaker per language (the first
# listed on its model card, preferring its studio "salt_" voices)
ORPHEUS = {"ach": "salt_ach_0001", "afr": "slr32_afr_0009", "eng": "salt_eng_0001",
           "ewe": "slr129_ewe_0001", "hau": "waxal_hau_0004", "ibo": "waxal_ibo_0003",
           "kik": "waxal_kik_0003", "kin": "bateesa_kin_0001", "lin": "slr129_lin_0001",
           "lug": "salt_lug_0001", "swh": "waxal_swa_0006", "xho": "slr32_xho_0012",
           "yor": "waxal_yor_0002"}


def run_orpheus(by_lang, out_root: Path, device: str, batch: int = 48) -> None:
    """Prompt format and sampling as on the model card (temperature 0.6, top-p 0.95, repetition
    penalty 1.1); batched with left padding."""
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = "Sunbird/orpheus-3b-tts-multilingual"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(device).eval()
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    EOT, SOS, EOS, SOH, EOH, PAD, LO = 128009, 128257, 128258, 128259, 128260, 128263, 128266

    def decode(ids: list[int]) -> np.ndarray:
        ids = torch.tensor(ids)
        sos = (ids == SOS).nonzero(as_tuple=True)[0]
        if len(sos):
            ids = ids[sos[-1] + 1:]
        a = ids[(ids >= LO) & (ids < LO + 7 * 4096)] - LO
        a = a[: len(a) // 7 * 7].view(-1, 7)
        if not len(a):
            return np.zeros(12000, dtype=np.float32)
        off = torch.arange(7) * 4096
        a = (a - off).clamp(0, 4095)
        l1, l2 = a[:, 0], a[:, [1, 4]].reshape(-1)
        l3 = a[:, [2, 3, 5, 6]].reshape(-1)
        with torch.no_grad():
            w = snac.decode([x.unsqueeze(0).to(device) for x in (l1, l2, l3)])
        return w.squeeze().float().cpu().numpy()

    for lang, rows in by_lang.items():
        if lang not in ORPHEUS:
            continue
        todo = [r for r in rows if not (out_root / lang / f"{r['id']}.wav").exists()]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            prompts = [[SOH] + tok.encode(f"{ORPHEUS[lang]}: {r['text']}", add_special_tokens=True)
                       + [EOT, EOH] for r in chunk]
            n = max(map(len, prompts))
            ids = torch.tensor([[PAD] * (n - len(p)) + p for p in prompts], device=device)
            mask = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device=device)
            torch.manual_seed(0)
            with torch.no_grad():
                gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5600,
                                     do_sample=True, temperature=0.6, top_p=0.95,
                                     repetition_penalty=1.1, eos_token_id=EOS, pad_token_id=PAD)
            for r, g in zip(chunk, gen[:, n:].tolist()):
                out = out_root / lang / f"{r['id']}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, decode(g), 24000)
        print(f"orpheus {lang}: {len(rows)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path)
    ap.add_argument("voices", type=Path)
    ap.add_argument("--system", required=True, choices=["mms", "omnivoice", "orpheus"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--voice-texts", type=Path, default=None,
                    help="OmniVoice prompt transcripts from transcribe_voices.py")
    a = ap.parse_args()
    by_lang = defaultdict(list)
    for line in open(a.bench / "testset.jsonl"):
        r = json.loads(line)
        by_lang[r["lang"]].append(r)
    out_root = a.bench / "synth" / a.system
    if a.system == "mms":
        run_mms(by_lang, out_root, a.device)
    elif a.system == "orpheus":
        run_orpheus(by_lang, out_root, a.device)
    else:
        run_omnivoice(by_lang, out_root, a.voices, a.device, a.voice_texts)


if __name__ == "__main__":
    main()
