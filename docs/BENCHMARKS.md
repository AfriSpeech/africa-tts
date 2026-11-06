# Benchmarks

## Speed

Measured with [`benchmarks/speed.py`](../benchmarks/speed.py) on a dedicated 4-vCPU Modal CPU
container ([`benchmarks/speed_modal.py`](../benchmarks/speed_modal.py)): 8 sentences in Twi, Hausa,
Yoruba, Swahili, Zulu, Amharic, Ewe and English, each language's default voice, after a warm-up
(voice prompts encoded once and cached). The Python package runs PyTorch on one thread
(pocket-tts's default); sherpa-onnx uses int8 models on four threads.

| Model and runtime | × real time | Real-time factor | Time to first audio (streaming, median) |
|---|---:|---:|---:|
| **Turbo**, sherpa-onnx int8 | **2.52** | 0.40 | — |
| **Turbo**, Python package | **2.38** | 0.42 | **0.17 s** |
| 24 layers unguided, sherpa-onnx int8 | 0.82 | 1.22 | — |
| 24 layers unguided, Python package | 0.90 | 1.11 | 0.52 s |
| 24 layers guided, Python package | 0.47 | 2.11 | 0.72 s |

"× real time" is seconds of audio per second of compute: above 1, audio is generated faster than
it plays. Raw results: [benchmarks/results/speed_modal_4vcpu.json](../benchmarks/results/speed_modal_4vcpu.json).
On the API server, each worker runs one request at a time, so throughput scales with
workers × cores ([server/](../server/README.md#scaling)).

Run it on your machine:

```bash
pip install "africa-tts @ git+https://github.com/AfriSpeech/africa-tts" sherpa-onnx soundfile
python benchmarks/speed.py --threads 4
```

## Intelligibility

Character error rate of [Omnilingual ASR](https://huggingface.co/facebook/omniASR-LLM-7B)
transcripts of each system's speech, on held-out sentences in 46 languages, against MMS-TTS and
OmniVoice: running; results will be published here and on the model cards.
