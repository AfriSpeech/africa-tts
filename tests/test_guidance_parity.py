"""Guided pocket-tts inference (recipes/africa/guidance.py) == the training code's guided sampler.

Generates with training's TrainableTTS.generate(cfg_coef=2.0), recording each frame's noise,
then replays the same noise through GuidedTTSModel (pocket-tts's own pipeline) from the same
voice latents and text, and compares the latents frame by frame.

    python tests/test_guidance_parity.py TRAIN_CONFIG CKPT MODEL_DIR VOICE_WAV "ipa text."
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipes" / "africa"))
import yaml
from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.models import tts_model as tm

from guidance import GuidedTTSModel  # noqa: E402
from training.args import load_args
from training.modules.builders import build_models


def main() -> None:
    train_cfg, ckpt, model_dir, voice_wav, text = sys.argv[1:6]
    torch.manual_seed(0)
    cfg, temp, frames = 2.0, 0.3, 40

    # pocket-tts model (released folder, local weights)
    c = yaml.safe_load((Path(model_dir) / "config.yaml").read_text())
    c["weights_path"] = str(Path(model_dir) / "model.safetensors")
    c["flow_lm"]["lookup_table"]["tokenizer_path"] = str(Path(model_dir) / "tokenizer.json")
    tmp = Path(model_dir) / "_local_cfg.yaml"
    tmp.write_text(yaml.safe_dump(c, allow_unicode=True))
    pocket = GuidedTTSModel.wrap(TTSModel.load_model(config=str(tmp), temp=temp), cfg)
    tmp.unlink()

    # the same voice latents for both: pocket's own prompt processing (resample, end_on_pause)
    audio, sr = audio_read(Path(voice_wav))
    audio = convert_audio(audio, sr, pocket.config.mimi.sample_rate, 1)  # pocket trims it itself
    trimmed = tm.end_on_pause(audio, pocket.config.mimi.sample_rate)
    voice_latents = pocket.mimi.encode_to_latent(trimmed.unsqueeze(0))[0].float()  # [Tv, C]

    # training sampler, recording noise
    args = load_args(train_cfg)
    model, _, _ = build_models(args)
    p = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = dict(p["model"])
    state.update(p.get("ema") or {})
    model.load_state_dict(state)
    model.eval()
    prepared = pocket.flow_lm.conditioner.prepare(text)  # same tokens as pocket will use
    noises: list[torch.Tensor] = []
    ref = model.generate([prepared[0]], [voice_latents], max_frames=frames, temp=temp,
                         cfg_coef=cfg, eos_threshold=1e9, noises=noises)[0]
    fl = model.flow_lm
    ref_norm = (ref - fl.emb_mean) / fl.emb_std  # back to the normalised space pocket emits

    # pocket guided pipeline with the same noise
    replay = iter(noises)
    orig = torch.nn.init.normal_

    def fake_normal(t: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        with torch.no_grad():
            return t.copy_(next(replay).view_as(t))

    # prefills (voice, text) also sample a discarded latent: replay noise only for the frames
    st = pocket.get_state_for_audio_prompt(audio)
    pocket._expand_kv_cache(st, sequence_length=2000)
    pocket._run_flow_lm_and_increment_step(model_state=st, text_tokens=prepared)
    torch.nn.init.normal_ = fake_normal
    try:
        x = torch.full((1, 1, fl.ldim), float("nan"))
        out = []
        for _ in range(len(noises)):
            lat, _ = pocket._run_flow_lm_and_increment_step(model_state=st, backbone_input_latents=x)
            out.append(lat[0, 0])
            x = lat
    finally:
        torch.nn.init.normal_ = orig
    got = torch.stack(out)
    n = min(len(got), len(ref_norm))
    diff = (got[:n] - ref_norm[:n]).abs()
    print(f"frames compared: {n}; max abs diff {diff.max():.2e}; "
          f"per-frame max (first, last): {diff[0].max():.2e}, {diff[-1].max():.2e}")
    assert n >= 30 and diff[0].max() < 1e-3, "guided pocket path does not match training"
    print("OK")


if __name__ == "__main__":
    main()
