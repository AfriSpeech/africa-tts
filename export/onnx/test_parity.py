#!/usr/bin/env python3
"""Numerical parity: exported ONNX graphs vs the PyTorch pocket_tts model (fp32, deterministic).

    python test_parity.py --model-dir MODEL --onnx-dir OUT --voice voice.wav --text "<IPA>" [--int8]

The ONNX side mirrors sherpa-onnx's Pocket TTS loop (encoder -> lm_main(voice) ->
lm_main(text) -> per frame lm_main + one LSD step of lm_flow -> chunked decoder), with zero
noise (temperature 0) so both sides are deterministic.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from export_onnx import load_tts
from pocket_tts.modules.stateful_module import increment_steps, init_states

STEPS = 40
CHUNK = 15


def ort_session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 8
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def zero_state(sess, first):
    out = {}
    for i in sess.get_inputs()[first:]:
        dt = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(bool)": np.bool_}[i.type]
        out[i.name] = np.zeros([d if isinstance(d, int) else 1 for d in i.shape], dt)
    return out


def maxdiff(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.abs(a - b).max()), float(np.abs(b).max())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    sfx = ".int8.onnx" if a.int8 else ".onnx"

    tts = load_tts(a.model_dir)
    fl = tts.flow_lm
    tts.temp, tts.noise_clamp = 0.0, None
    wav, sr = sf.read(a.voice, dtype="float32", always_2d=True)
    assert sr == 24000
    wav = wav[:, 0]

    # ---- PyTorch reference ----
    def pt_prompt(w):
        cond = tts._encode_audio(torch.from_numpy(w)[None, None])
        return torch.cat([fl.bos_before_voice, cond], dim=1) if fl.insert_bos_before_voice else cond

    tokens = fl.conditioner.prepare(a.text)  # [1, T]
    eos_logits = []
    hook = fl.out_eos.register_forward_hook(lambda m, i, o: eos_logits.append(o.clone()))
    state = init_states(fl, batch_size=1, sequence_length=1000)
    prompt = pt_prompt(wav)
    tts._run_flow_lm_and_increment_step(model_state=state, audio_conditioning=prompt)
    tts._run_flow_lm_and_increment_step(model_state=state, text_tokens=tokens)
    eos_logits.clear()
    prev = torch.full((1, 1, fl.ldim), float("nan"))
    pt_lat = []
    for _ in range(STEPS):
        lat, _ = tts._run_flow_lm_and_increment_step(model_state=state, backbone_input_latents=prev)
        pt_lat.append(lat[0, 0].numpy())
        prev = lat
    hook.remove()
    pt_eos = torch.cat(eos_logits).numpy()
    pt_lat = np.stack(pt_lat)
    mstate = init_states(tts.mimi, batch_size=1, sequence_length=STEPS * 16 + 64)
    pt_audio = []
    lat_t = torch.from_numpy(pt_lat)[None]
    for s in range(0, STEPS, CHUNK):
        chunk = lat_t[:, s:s + CHUNK]
        pt_audio.append(tts.mimi.decode_from_latent(chunk * fl.emb_std + fl.emb_mean, mstate)[0, 0].numpy())
        increment_steps(tts.mimi, mstate, increment=16 * chunk.shape[1])
    pt_audio = np.concatenate(pt_audio)

    # ---- ONNX ----
    d = a.onnx_dir
    enc = ort_session(d / ("encoder" + sfx))
    txt = ort_session(d / ("text_conditioner" + sfx))
    main_ = ort_session(d / ("lm_main" + sfx))
    flow = ort_session(d / ("lm_flow" + sfx))
    dec = ort_session(d / ("decoder" + sfx))
    res = {}

    for name, w in [("encoder 5.0s", wav), ("encoder 3.37s (odd length)", wav[: int(3.37 * 24000) + 11])]:
        o = enc.run(None, {"audio": w[None, None]})[0]
        res[name] = maxdiff(o, pt_prompt(w).numpy())
    voice = enc.run(None, {"audio": wav[None, None]})[0]
    temb = txt.run(None, {"token_ids": tokens.numpy().astype(np.int64)})[0]
    res["text_conditioner"] = maxdiff(temb, fl.conditioner(tokens).numpy())

    st = zero_state(main_, 2)
    names_out = [o.name for o in main_.get_outputs()]

    def run_main(seq, emb):
        out = main_.run(None, {"sequence": seq, "text_embeddings": emb, **st})
        for k, v in zip(list(st), out[2:]):
            st[k] = v
        return out[0], out[1]

    empty_seq = np.zeros((1, 0, fl.ldim), np.float32)
    empty_txt = np.zeros((1, 0, fl.dim), np.float32)
    run_main(empty_seq, voice)
    run_main(empty_seq, temb)
    cur = np.full((1, 1, fl.ldim), np.nan, np.float32)
    on_lat, on_eos = [], []
    for _ in range(STEPS):
        c, eos = run_main(cur, empty_txt)
        x = np.zeros((1, fl.ldim), np.float32)
        x = x + flow.run(None, {"c": c, "s": np.zeros((1, 1), np.float32), "t": np.ones((1, 1), np.float32), "x": x})[0]
        on_lat.append(x[0])
        on_eos.append(eos[0])
        cur = x[None]
    on_lat = np.stack(on_lat)
    res[f"lm_main+lm_flow latents ({STEPS} AR steps, own feedback)"] = maxdiff(on_lat, pt_lat)
    res["lm_main eos_logit"] = maxdiff(np.stack(on_eos), pt_eos)
    per_step = np.abs(on_lat - pt_lat).max(1)
    print("latent max diff per step:", " ".join(f"{i}:{per_step[i]:.1e}" for i in [0, 1, 2, 5, 10, 20, 30, STEPS - 1]))

    # decoder on the *reference* latents, chunked like sherpa (chunk_size 15)
    dst = zero_state(dec, 1)
    on_audio = []
    for s in range(0, STEPS, CHUNK):
        out = dec.run(None, {"latent": pt_lat[None, s:s + CHUNK].astype(np.float32), **dst})
        for k, v in zip(list(dst), out[1:]):
            dst[k] = v
        on_audio.append(out[0][0, 0])
    on_audio = np.concatenate(on_audio)
    res["decoder audio (chunked, same latents)"] = maxdiff(on_audio, pt_audio)
    snr = 10 * np.log10((pt_audio ** 2).sum() / max(((on_audio - pt_audio) ** 2).sum(), 1e-20))

    print(f"{'int8' if a.int8 else 'fp32'} parity (max |onnx - torch|, max |torch|):")
    for k, (m, ref) in res.items():
        print(f"  {k:55s} {m:.3e}   (ref max {ref:.3f})")
    print(f"  decoder SNR vs torch: {snr:.1f} dB")


if __name__ == "__main__":
    main()
