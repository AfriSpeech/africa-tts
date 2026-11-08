#!/usr/bin/env python3
"""Export a Pocket TTS model folder to the ONNX file set sherpa-onnx's Pocket TTS backend loads.

    python export_onnx.py --model-dir release/africa-tts --out-dir out/africa-tts [--quantize]

Input folder: config.yaml + model.safetensors + tokenizer.json (a pocket-tts model, e.g.
AfriSpeech/africa-tts or africa-tts-base; any number of FlowLM layers). Output:

    lm_main.onnx           FlowLM backbone, stateful (KV cache in/out)
    lm_flow.onnx           flow head, one LSD step: x + dt * flow_dir
    encoder.onnx           reference audio -> voice conditioning (BOS prepended, see below)
    decoder.onnx           Mimi decoder, stateful, latents -> audio
    text_conditioner.onnx  token ids -> text embeddings
    *.int8.onnx            MatMul-only dynamic int8 versions (--quantize)
    vocab.json, token_scores.json   tokenizer for sherpa-onnx's SentencePiece re-implementation

Differences from the reference exporter:
- Ported to the current pocket_tts package (linear KV caches, new attention module), with
  functional re-implementations of the streaming modules: every KV cache is a fixed-capacity
  shift buffer (the newest `capacity` keys), so all state tensors have static shapes, which
  sherpa-onnx needs to zero-initialise them. FlowLM capacity 1000 positions (voice + text +
  generated frames; sherpa caps voice at 10 s and frames at 500), Mimi decoder capacity = its
  attention context (250).
- Models with `insert_bos_before_voice` (the 2026 pocket-tts family, incl. Africa TTS) prepend a
  learned BOS vector to the voice prompt. sherpa-onnx feeds the encoder output straight into
  lm_main, so the BOS is baked into encoder.onnx's output.
- The tokenizer comes from a HF `tokenizers` Unigram tokenizer.json (or a SentencePiece .model).

Credits: based on pocket-tts-onnx-export by KevinAHM (https://huggingface.co/KevinAHM/pocket-tts-onnx)
as adapted for sherpa-onnx by csukuangfj (https://github.com/csukuangfj/pocket-tts-onnx-export),
MIT License, and on Kyutai's Pocket TTS (https://github.com/kyutai-labs/pocket-tts), MIT License.

MIT License. Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions: The above copyright notice
and this permission notice shall be included in all copies or substantial portions of the
Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from pocket_tts import TTSModel
from pocket_tts.modules.attention import StreamingMultiheadAttention
from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.stateful_module import StatefulModule
from torch import nn

LM_CAPACITY = 1000
OPSET = 14
MIMI_STEPS_PER_LATENT = 16  # encoder frame rate (200 Hz) / latent frame rate (12.5 Hz)


# --------------------------------------------------------------------------------------------
# Functional streaming modules. State tensors are replaced, never mutated in place, so the
# traced graph maps state_i inputs to out_state_i outputs.
# --------------------------------------------------------------------------------------------


def _attention_forward(self: StreamingMultiheadAttention, query, model_state, attn_mask=None):
    state = None if model_state is None else self.get_state(model_state)
    b, t, _ = query.shape
    d = self.dim_per_head
    projected = self.in_proj(query)
    q = projected[..., : self.attn_dim].reshape(b, t, self.num_heads, d)
    kv = projected[..., self.attn_dim :].reshape(b, t, 2, self.num_kv_heads, d)
    k, v = kv[:, :, 0], kv[:, :, 1]
    ar_t = torch.arange(t, device=query.device, dtype=torch.long)
    if state is None:  # stateless (encoder): plain causal attention over the input
        q, k = self.rope(q, k, offset=0)
        k_all, v_all, pos_q, pos_k = k, v, ar_t, ar_t
    else:
        offset = state["offset"][:1]  # long [1]
        q, k = self.rope(q, k, offset=offset)
        cache = state["cache"]  # [2, B, C, H_kv, D]: newest C keys/values (rotated)
        capacity = cache.shape[2]
        k_all = torch.cat([cache[0], k], dim=1)
        v_all = torch.cat([cache[1], v], dim=1)
        pos_q = offset + ar_t
        pos_k = offset - capacity + torch.arange(capacity + t, device=query.device, dtype=torch.long)
        state["cache"] = torch.stack([k_all[:, -capacity:], v_all[:, -capacity:]])
        state["offset"] = state["offset"] + t
    delta = pos_q[:, None] - pos_k[None, :]
    allowed = (pos_k[None, :] >= 0) & (delta >= 0)
    if self.context is not None:
        allowed = allowed & (delta < self.context)
    mask = torch.zeros(allowed.shape, dtype=q.dtype, device=q.device).masked_fill(
        ~allowed, float("-inf")
    )
    k_all = k_all.transpose(1, 2)
    v_all = v_all.transpose(1, 2)
    if self.num_kv_heads != self.num_heads:
        rep = self.num_heads // self.num_kv_heads
        k_all = k_all.repeat_interleave(rep, dim=1)
        v_all = v_all.repeat_interleave(rep, dim=1)
    x = F.scaled_dot_product_attention(q.transpose(1, 2), k_all, v_all, mask)
    x = x.transpose(1, 2).reshape(b, t, self.num_heads * d)
    return self.out_proj(x)


def _attention_init_state(self: StreamingMultiheadAttention, batch_size: int, sequence_length: int):
    capacity = self.context if self.context is not None else sequence_length
    w = self.in_proj.weight
    return dict(
        offset=torch.zeros(batch_size, dtype=torch.long),
        cache=torch.zeros(2, batch_size, capacity, self.num_kv_heads, self.dim_per_head, dtype=w.dtype),
    )


def _conv_forward(self: StreamingConv1d, x, model_state):
    tp = self._effective_kernel_size - self._stride
    if model_state is None:
        if tp:
            if self.pad_mode == "replicate":
                prev = x[..., :1].expand(-1, -1, tp)
            else:
                prev = torch.zeros_like(x[..., :1]).expand(-1, -1, tp)
            x = torch.cat([prev, x], dim=-1)
        return self.conv(x)
    assert self.pad_mode == "constant", "stateful replicate padding is not exported"
    state = self.get_state(model_state)
    if tp:
        x = torch.cat([state["previous"], x], dim=-1)
        state["previous"] = x[..., -tp:]
    return self.conv(x)


def _conv_init_state(self: StreamingConv1d, batch_size: int, sequence_length: int):
    tp = self._effective_kernel_size - self._stride
    return dict(previous=torch.zeros(batch_size, self.conv.in_channels, tp))


def _convtr_forward(self: StreamingConvTranspose1d, x, model_state):
    state = self.get_state(model_state)
    partial = state["partial"]
    y = self.convtr(x)
    pt = partial.shape[-1]
    if pt > 0:
        y = torch.cat([y[..., :pt] + partial, y[..., pt:]], dim=-1)
        for_partial = y[..., -pt:]
        if self.convtr.bias is not None:
            for_partial = for_partial - self.convtr.bias[:, None]
        state["partial"] = for_partial
        y = y[..., :-pt]
    return y


def _noop_increment(self, state, increment=1):
    pass


def patch_streaming_modules():
    StreamingMultiheadAttention.forward = _attention_forward
    StreamingMultiheadAttention.init_state = _attention_init_state
    StreamingMultiheadAttention.increment_step = _noop_increment  # done inside forward
    StreamingConv1d.forward = _conv_forward
    StreamingConv1d.init_state = _conv_init_state
    StreamingConvTranspose1d.forward = _convtr_forward


def make_state(module: nn.Module, sequence_length: int, prefixes: tuple[str, ...] = ("",)
               ) -> tuple[list[str], list[torch.Tensor]]:
    """Zero state for every StatefulModule (under `prefixes`), flattened in a fixed order."""
    names, tensors = [], []
    for name, m in module.named_modules():
        if isinstance(m, StatefulModule) and name.startswith(prefixes):
            for key, t in sorted(m.init_state(1, sequence_length).items()):
                names.append(f"{name}/{key}")
                tensors.append(t)
    return names, tensors


def unflatten(names: list[str], tensors) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    for n, t in zip(names, tensors):
        mod, key = n.rsplit("/", 1)
        state.setdefault(mod, {})[key] = t
    return state


def flatten(names: list[str], state) -> list[torch.Tensor]:
    return [state[n.rsplit("/", 1)[0]][n.rsplit("/", 1)[1]] for n in names]


# --------------------------------------------------------------------------------------------
# Wrappers (I/O names follow the reference exporter, which sherpa-onnx was written against)
# --------------------------------------------------------------------------------------------


class LmMain(nn.Module):
    """sequence [1,S,ldim] (NaN = BOS), text_embeddings [1,T,dim], state... ->
    conditioning [1,dim] (last position), eos_logit [1,1], new state..."""

    def __init__(self, flow_lm, state_names):
        super().__init__()
        self.flow_lm = flow_lm
        self.state_names = state_names

    def forward(self, sequence, text_embeddings, *state_flat):
        fl = self.flow_lm
        state = unflatten(self.state_names, state_flat)
        sequence = torch.where(torch.isnan(sequence), fl.bos_emb, sequence)
        # sherpa-onnx passes [1, 0, 1024] as empty text_embeddings during AR steps; slice to fl.dim
        text_embeddings = text_embeddings[:, :, :fl.dim]
        x = torch.cat([text_embeddings, fl.input_linear(sequence)], dim=1)
        x = fl.transformer(x, state)
        if fl.out_norm:
            x = fl.out_norm(x)
        last = x[:, -1]
        if fl.z_proj is not None:
            last = fl.z_proj(last)
        last = last.to(torch.float32)
        return (last, fl.out_eos(last), *flatten(self.state_names, state))


class LmFlow(nn.Module):
    def __init__(self, flow_lm):
        super().__init__()
        self.flow_net = flow_lm.flow_net

    def forward(self, c, s, t, x):
        return self.flow_net(c, s, t, x)


class Encoder(nn.Module):
    """audio [1,1,N] at 24 kHz -> voice conditioning [1, 1 + frames, dim]."""

    def __init__(self, tts: TTSModel):
        super().__init__()
        self.mimi = tts.mimi
        self.frame_size = tts.mimi.frame_size
        self.register_buffer("speaker_proj_weight", tts.flow_lm.speaker_proj_weight.detach().clone())
        bos = getattr(tts.flow_lm, "bos_before_voice", None) if tts.flow_lm.insert_bos_before_voice else None
        self.register_buffer("bos", None if bos is None else bos.detach().clone())

    def forward(self, audio):
        m = self.mimi
        # pad to a whole number of frames (dynamic; pad_for_conv1d's math.ceil would be traced
        # as a constant)
        n = audio.shape[-1]
        padded = F.pad(audio, (0, self.frame_size))
        target = ((n + self.frame_size - 1) // self.frame_size) * self.frame_size
        audio = padded[..., :target]
        emb = m.encoder(audio, model_state=None)
        (emb,) = m.encoder_transformer(emb, model_state=None)
        emb = m._to_framerate(emb)
        latents = emb.transpose(-1, -2).to(torch.float32)
        cond = F.linear(latents, self.speaker_proj_weight)
        if self.bos is not None:
            cond = torch.cat([self.bos.expand(cond.shape[0], -1, -1), cond], dim=1)
        return cond


class TextConditioner(nn.Module):
    def __init__(self, conditioner):
        super().__init__()
        self.conditioner = conditioner

    def forward(self, token_ids):
        return self.conditioner(token_ids)


class Decoder(nn.Module):
    """latent [1,T,ldim] (normalised, as produced by lm_flow), state... -> audio [1,1,T*1920]."""

    def __init__(self, tts: TTSModel, state_names):
        super().__init__()
        self.mimi = tts.mimi
        self.state_names = state_names
        self.register_buffer("emb_std", tts.flow_lm.emb_std.detach().clone())
        self.register_buffer("emb_mean", tts.flow_lm.emb_mean.detach().clone())

    def forward(self, latent, *state_flat):
        state = unflatten(self.state_names, state_flat)
        audio = self.mimi.decode_from_latent(latent * self.emb_std + self.emb_mean, state)
        return (audio, *flatten(self.state_names, state))


# --------------------------------------------------------------------------------------------


def load_tts(model_dir: Path) -> TTSModel:
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
    cfg["weights_path"] = str((model_dir / "model.safetensors").resolve())
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str((model_dir / "tokenizer.json").resolve())
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return TTSModel.load_model(config=f.name).cpu().eval()


def export_tokenizer(model_dir: Path, out_dir: Path):
    tok = model_dir / "tokenizer.json"
    if tok.exists():
        model = json.loads(tok.read_text(encoding="utf-8"))["model"]
        assert model["type"] == "Unigram", model["type"]
        pieces = [(p, float(s)) for p, s in model["vocab"]]
    else:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
        pieces = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())]
    vocab = {p: i for i, (p, _) in enumerate(pieces)}
    scores = {p: s for p, s in pieces}
    assert len(vocab) == len(pieces), "duplicate pieces"
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "token_scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"tokenizer: {len(vocab)} pieces")


def _export(module, args, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(module, args, str(path), input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=OPSET, dynamo=False, do_constant_folding=True)
    print(f"exported {path.name} ({path.stat().st_size / 2**20:.1f} MB)")


def quantize(out_dir: Path):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for name in ["lm_main", "lm_flow", "encoder", "decoder", "text_conditioner"]:
        src, dst = out_dir / f"{name}.onnx", out_dir / f"{name}.int8.onnx"
        tmp = out_dir / f"{name}.shape.onnx"
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(str(src))), str(tmp))
        quantize_dynamic(str(tmp), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                         extra_options={"ForceQuantizeNoType": True, "DefaultTensorType": 1})
        tmp.unlink()
        print(f"quantized {dst.name} ({dst.stat().st_size / 2**20:.1f} MB)")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--quantize", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    tts = load_tts(a.model_dir)
    patch_streaming_modules()
    fl = tts.flow_lm
    print(f"FlowLM: {len(fl.transformer.layers)} layers, dim {fl.dim}, ldim {fl.ldim}, "
          f"bos_before_voice={fl.insert_bos_before_voice}")

    export_tokenizer(a.model_dir, a.out_dir)

    _export(TextConditioner(fl.conditioner), (torch.randint(0, 100, (1, 12)),),
            a.out_dir / "text_conditioner.onnx", ["token_ids"], ["embeddings"],
            {"token_ids": {1: "num_tokens"}, "embeddings": {1: "num_tokens"}})

    _export(Encoder(tts), (torch.randn(1, 1, 24000 * 3 + 517),), a.out_dir / "encoder.onnx",
            ["audio"], ["latents"], {"audio": {2: "num_samples"}, "latents": {1: "num_frames"}})

    names, state = make_state(fl, LM_CAPACITY)
    _export(LmMain(fl, names), (torch.randn(1, 1, fl.ldim), torch.randn(1, 3, fl.dim), *state),
            a.out_dir / "lm_main.onnx",
            ["sequence", "text_embeddings"] + [f"state_{i}" for i in range(len(state))],
            ["conditioning", "eos_logit"] + [f"out_state_{i}" for i in range(len(state))],
            {"sequence": {1: "seq_len"}, "text_embeddings": {1: "text_len", 2: "text_dim"}})

    cond_dim = fl.flow_net.cond_embed.in_features
    _export(LmFlow(fl), (torch.randn(1, cond_dim), torch.zeros(1, 1), torch.ones(1, 1), torch.randn(1, fl.ldim)),
            a.out_dir / "lm_flow.onnx", ["c", "s", "t", "x"], ["flow_dir"],
            {k: {0: "batch"} for k in ["c", "s", "t", "x", "flow_dir"]})

    # decoder side only; attention capacity = its context
    names, state = make_state(tts.mimi, 0, ("upsample.", "decoder_transformer.", "decoder."))
    _export(Decoder(tts, names), (torch.randn(1, 2, fl.ldim), *state), a.out_dir / "decoder.onnx",
            ["latent"] + [f"state_{i}" for i in range(len(state))],
            ["audio_frame"] + [f"out_state_{i}" for i in range(len(state))],
            {"latent": {1: "num_latents"}, "audio_frame": {2: "num_samples"}})

    if a.quantize:
        quantize(a.out_dir)


if __name__ == "__main__":
    main()
