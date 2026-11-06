"""Classifier-free guidance (CFG) for pocket-tts inference.

pocket-tts generates without guidance. The 24-layer Africa TTS release was listened to with
cfg 2.0 (also the output Turbo was distilled from), so this adds guidance to pocket-tts's own
generation pipeline, exactly as the training code samples:

    conditioned branch   [BOS-before-voice, voice, text, audio...]   (the usual voice state)
    null branch          [BOS-before-voice,               audio...]   (no voice, no text)
    z = z_null + cfg * (z_cond - z_null)          at every frame, on the backbone output;
    the flow head and the EOS head then read z.

Both branches are fed the same generated frames. The null branch's KV cache travels inside the
voice state (key NULL_KEY), so pocket-tts's chunking, state copies, streaming decoder and EOS
logic are reused unchanged. Each frame runs the backbone twice: about 2x slower.

Turbo has guidance distilled into it: use it with cfg 1 (no guidance).
"""

from functools import partial
from pathlib import Path

import torch
from pocket_tts import TTSModel
from pocket_tts.models.flow_lm import drifting_decode, lsd_decode, ot_decode
from pocket_tts.modules.stateful_module import ModelState, increment_steps, init_states

NULL_KEY = "__africa_tts_cfg_null__"
_DECODE = {"lsd": lsd_decode, "flow_matching": ot_decode, "drifting": drifting_decode}


class GuidedTTSModel(TTSModel):
    """A pocket-tts TTSModel that samples with classifier-free guidance when cfg_coef != 1."""

    cfg_coef: float = 1.0

    @classmethod
    def wrap(cls, model: TTSModel, cfg_coef: float) -> "GuidedTTSModel":
        """Turn a loaded TTSModel into a GuidedTTSModel in place (no weight copy)."""
        if not model.flow_lm.insert_bos_before_voice:
            raise ValueError("guidance needs a model with insert_bos_before_voice (the null prefix)")
        model.__class__ = cls
        model.cfg_coef = float(cfg_coef)  # ty: ignore[invalid-assignment]
        return model  # ty: ignore[invalid-return-type]

    @torch.no_grad
    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> ModelState:
        state = super().get_state_for_audio_prompt(audio_conditioning, truncate)
        if self.cfg_coef != 1.0:
            null = init_states(self.flow_lm, batch_size=1, sequence_length=1)
            super()._run_flow_lm_and_increment_step(
                model_state=null, audio_conditioning=self.flow_lm.bos_before_voice
            )
            state[NULL_KEY] = null  # ty: ignore[invalid-assignment]
        return state

    def _expand_kv_cache(self, model_state: ModelState, sequence_length: int):
        super()._expand_kv_cache(model_state, sequence_length)
        null = model_state.get(NULL_KEY)
        if null is not None:
            super()._expand_kv_cache(null, sequence_length)  # ty: ignore[invalid-argument-type]

    def _run_flow_lm_and_increment_step(
        self,
        model_state: ModelState,
        text_tokens: torch.Tensor | None = None,
        backbone_input_latents: torch.Tensor | None = None,
        audio_conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        null = model_state.get(NULL_KEY)
        has_text = text_tokens is not None and text_tokens.shape[1] > 0
        n_lat = 0 if backbone_input_latents is None else backbone_input_latents.shape[1]
        if null is None or self.cfg_coef == 1.0 or n_lat == 0:
            # Prefills (voice, text) go to the conditioned branch only: the null branch's
            # prefix is just BOS-before-voice, fed when the voice state was made.
            return super()._run_flow_lm_and_increment_step(
                model_state, text_tokens, backbone_input_latents, audio_conditioning
            )
        if has_text or audio_conditioning is not None or n_lat != 1:
            raise NotImplementedError("guidance does not support teacher_force_previous_chunk")
        assert backbone_input_latents is not None
        fl = self.flow_lm
        seq = torch.where(torch.isnan(backbone_input_latents), fl.bos_emb, backbone_input_latents)
        inp = fl.input_linear(seq)
        empty = inp.new_empty((inp.shape[0], 0, fl.dim))
        z_cond = fl.backbone(inp, empty, seq, model_state=model_state)[:, -1].float()
        z_null = fl.backbone(inp, empty, seq, model_state=null)[:, -1].float()  # ty: ignore[invalid-argument-type]
        increment_steps(fl, model_state, increment=1)
        increment_steps(fl, null, increment=1)  # ty: ignore[invalid-argument-type]
        z = z_null + self.cfg_coef * (z_cond - z_null)

        is_eos = fl.out_eos(z) > self.eos_threshold
        noise = torch.empty(z.shape[:-1] + (fl.ldim,), dtype=z.dtype, device=z.device)
        std = self.temp**0.5
        if self.noise_clamp is None:
            torch.nn.init.normal_(noise, mean=0.0, std=std)
        else:
            torch.nn.init.trunc_normal_(noise, mean=0.0, std=std, a=-self.noise_clamp,
                                        b=self.noise_clamp)
        latent = _DECODE[fl.flow_type](partial(fl.flow_net, z), noise, self.sampler_decode_steps)
        return latent[:, None, :], is_eos
