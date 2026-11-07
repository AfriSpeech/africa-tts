"""Transcribe the default voice clips with Omnilingual ASR, for baselines that need the prompt's
transcript (OmniVoice). The JW-recording voices store only a fragment of their transcript (the
clip was cut at word boundaries for Africa TTS, which clones from audio alone), so a cloning model
given that text is mis-prompted. Female-speech voices keep their human transcripts.

Usage: python transcribe_voices.py VOICES_DIR OUT.json
Writes {lang: {"text": transcript, "from": "asr" | "human"}} for every voice in voices.json.
"""

import json
import sys
from pathlib import Path

import soundfile as sf
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from transcribe import asr_lang


def main() -> None:
    voices, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads((voices / "voices.json").read_text())
    jw = [lg for lg, m in meta.items() if m.get("source") == "afrispeech"]
    pipe = ASRInferencePipeline(model_card="omniASR_LLM_7B_v2")
    res = {lg: {"text": m["text"], "from": "human"} for lg, m in meta.items() if lg not in jw}
    known = [lg for lg in jw if asr_lang(lg, None)]
    unknown = [lg for lg in jw if not asr_lang(lg, None)]

    def load(lg):
        wav, sr = sf.read(voices / f"{lg}.wav", dtype="float32", always_2d=True)
        return {"waveform": wav.mean(1), "sample_rate": sr}

    if known:
        for lg, h in zip(known, pipe.transcribe([load(lg) for lg in known],
                                               lang=[asr_lang(lg, None) for lg in known], batch_size=8)):
            res[lg] = {"text": h, "from": "asr"}
    if unknown:
        for lg, h in zip(unknown, pipe.transcribe([load(lg) for lg in unknown], batch_size=8)):
            res[lg] = {"text": h, "from": "asr"}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for lg in jw:
        print(lg, repr(meta[lg]["text"][:40]), "->", repr(res[lg]["text"][:80]))


if __name__ == "__main__":
    main()
