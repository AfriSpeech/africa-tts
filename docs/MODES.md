# Three ways to run Africa TTS

| Mode | How it works | Compute per audio frame | Runs in |
|---|---|---|---|
| **Africa TTS Turbo** | 6-layer model distilled from the guided 24-layer model, so guidance is built into its weights; one pass per frame | 6 layers | Python (`AfricaTTS.load()`, **the package default**), sherpa-onnx (server, on device) |
| **Africa TTS, guided** | 24-layer model with classifier-free guidance 2.0: every frame runs the backbone twice (with and without the text and voice) and pushes the output towards the conditioned one | 2 × 24 layers | Python (`AfricaTTS.load("africa-tts")`) |
| **Africa TTS, unguided** | the same 24-layer model, one pass per frame | 24 layers | Python (`AfricaTTS.load("africa-tts", cfg=1.0)`), sherpa-onnx |

The package loads **Turbo** unless you name a model (`AfricaTTS.load()`, or `africa-tts` without
`--model`); loading `africa-tts` uses guidance 2.0 unless you pass `cfg=1.0`.

The modes trade compute for different things, not along one quality scale: which sounds best
can differ by language and listener. Compare them on the
[samples page](https://huggingface.co/spaces/AfriSpeech/africa-tts-samples) and in the
[benchmark](BENCHMARKS.md).

## Guidance

Guidance is implemented in [africa_tts/guidance.py](../africa_tts/guidance.py) on top of
pocket-tts's own generation, and matches the training code's guided sampler to float precision
([tests/test_guidance_parity.py](../tests/test_guidance_parity.py)). sherpa-onnx cannot run
guidance (it needs the second pass), so on a server or device the choice is the unguided
24-layer model or Turbo.
