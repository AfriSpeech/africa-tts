"""PyTorch Dataset loader for African Speech TTS (aligned, IPA).

Loads audio clips directly by slicing their start and duration offsets from the container files.
Can be used with any PyTorch-based TTS or ASR training pipeline.

Example:
    from dataset import AfricanTTSDataset
    from torch.utils.data import DataLoader

    # Load all languages (or filter to specific languages)
    ds = AfricanTTSDataset(".", split="train", languages=["twi", "yor"])
    print(f"Loaded {len(ds)} clips")

    item = ds[0]
    print(item["language"], item["text"], item["audio"].shape)
"""

import json
from pathlib import Path
from typing import Sequence
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class AfricanTTSDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        languages: Sequence[str] | str | None = None,
        target_sr: int = 24000,
    ) -> None:
        self.root = Path(root)
        self.target_sr = target_sr
        manifest = self.root / f"{split}.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        if isinstance(languages, str):
            languages = [languages]
        allowed = {lang.lower() for lang in languages} if languages else None

        self.rows = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if allowed is None or r.get("g2p", "").lower() in allowed or r.get("iso", "").lower() in allowed:
                    self.rows.append(r)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        wav_path = str(self.root / row["path"])
        info = sf.info(wav_path)
        sr = info.samplerate

        start_frame = int(row["start"] * sr)
        frame_count = int(row["duration"] * sr)

        audio, _ = sf.read(
            wav_path,
            start=start_frame,
            frames=frame_count,
            dtype="float32",
            always_2d=True,
        )
        audio = audio.mean(1)

        # Resample to target_sr if needed
        if self.target_sr and sr != self.target_sr:
            from scipy.signal import resample_poly
            g = np.gcd(sr, self.target_sr)
            audio = resample_poly(audio, self.target_sr // g, sr // g).astype(np.float32)
            out_sr = self.target_sr
        else:
            out_sr = sr

        clip_id = str(row.get("g2p") or "clip") + "_" + str(row.get("subset") or "sub") + "_" + str(idx)
        return {
            "id": clip_id,
            "audio": torch.from_numpy(audio),
            "sample_rate": out_sr,
            "text": row.get("transcript_orig", row.get("text", "")),
            "ipa": row.get("transcript", ""),
            "language": row.get("language", ""),
            "iso": row.get("iso", ""),
            "g2p": row.get("g2p", ""),
            "country": row.get("country", ""),
            "duration": len(audio) / out_sr,
            "words": row.get("words", []),
        }
