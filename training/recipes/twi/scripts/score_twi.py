"""Score manifest rows for filtering raw multi-speaker podcast audio.

Adds to each row:
  spk_sim  cosine similarity (microsoft/wavlm-base-plus-sv) between the clip's first
           min(PROMPT_SEC, duration/2) seconds and the rest. Training uses the start of
           a clip as the voice prompt for the rest, so a low value means a speaker
           change (host/caller, overlapping talk) that would teach the wrong voice.
  utmos    UTMOS estimate on a centred CROP_SEC crop (or the whole clip if shorter).
           Low values flag noise, music beds, phone audio and clipping.

Resumable: rows already in the output (same order as the input) are skipped.

Usage:
  python data/score_twi.py data/twi_all/manifests/train_twi.jsonl data/twi_all/scores/train_twi.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

SR = 16_000
PROMPT_SEC = 4.0
CROP_SEC = 6.0


class Rows(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        try:
            info_sr = sf.info(r["path"]).samplerate
            start = int(round(float(r.get("start", 0.0)) * info_sr))
            n = int(round(float(r["duration"]) * info_sr))
            wav, sr = sf.read(r["path"], start=start, frames=n, dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            if sr != SR:
                g = int(np.gcd(sr, SR))
                wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
            if len(wav) < SR:  # < 1 s: nothing meaningful to score
                return i, None
            return i, wav
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING {r['path']}@{r.get('start')}: {exc}", file=sys.stderr, flush=True)
            return i, None


def collate(batch):
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_jsonl")
    ap.add_argument("output_jsonl")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.input_jsonl) if line.strip()]
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for line in open(out) if line.strip()) if out.exists() else 0
    todo = rows[done:]
    print(f"{len(rows):,} rows, {done:,} already scored, {len(todo):,} to go", flush=True)
    if not todo:
        return

    from transformers import AutoFeatureExtractor, WavLMForXVector
    from utmos_pytorch import UTMOSScoreTorch

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    spk = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()  # ty: ignore[invalid-argument-type]
    utmos = UTMOSScoreTorch(device=str(device))

    def embed(wavs: list[np.ndarray]) -> torch.Tensor:
        inputs = fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True).to(device)
        return spk(**inputs).embeddings

    def mos(wavs: list[np.ndarray]) -> list[float]:
        # Equal-length crops batch without padding (padding shifts UTMOS); shorter clips
        # are scored one at a time at their true length.
        crop = int(CROP_SEC * SR)
        scores: list[float] = [0.0] * len(wavs)
        full = [k for k, w in enumerate(wavs) if len(w) >= crop]
        if full:
            x = []
            for k in full:
                off = (len(wavs[k]) - crop) // 2
                x.append(wavs[k][off : off + crop])
            s = utmos.score(torch.from_numpy(np.stack(x))[:, None].to(device))
            for k, v in zip(full, s.flatten().tolist()):
                scores[k] = float(v)
        for k, w in enumerate(wavs):
            if len(w) < crop:
                scores[k] = float(utmos.score(torch.from_numpy(w)[None, None].to(device)))
        return scores

    loader = DataLoader(
        Rows(todo),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate,
        prefetch_factor=4,
    )
    n = 0
    with open(out, "a") as fout, torch.inference_mode():
        for batch in loader:
            ok = [(i, w) for i, w in batch if w is not None]
            sims: dict[int, float] = {}
            moses: dict[int, float] = {}
            if ok:
                heads, tails = [], []
                for _, w in ok:
                    cut = int(min(PROMPT_SEC, len(w) / SR / 2) * SR)
                    heads.append(w[:cut])
                    tails.append(w[cut:])
                e = embed(heads + tails)
                s = torch.nn.functional.cosine_similarity(e[: len(ok)], e[len(ok) :], dim=-1)
                m = mos([w for _, w in ok])
                for (i, _), sv, mv in zip(ok, s.tolist(), m):
                    sims[i], moses[i] = float(sv), mv
            for i, _ in batch:  # keep input order for resume
                r = dict(todo[i])
                r["spk_sim"] = round(sims[i], 4) if i in sims else None
                r["utmos"] = round(moses[i], 4) if i in moses else None
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += len(batch)
            if n % (args.batch_size * 200) < args.batch_size:
                fout.flush()
                print(f"scored {done + n:,}/{len(rows):,}", flush=True)

    scored = [json.loads(line) for line in open(out) if line.strip()]
    for key in ("spk_sim", "utmos"):
        v = np.array([r[key] for r in scored if r.get(key) is not None])
        if len(v):
            pct = np.percentile(v, [1, 5, 10, 25, 50, 75, 90])
            print(f"{key}: n={len(v):,} p1/5/10/25/50/75/90 = {np.round(pct, 3).tolist()}")
    print(f"unscorable rows: {sum(r.get('utmos') is None for r in scored):,}")


if __name__ == "__main__":
    main()
