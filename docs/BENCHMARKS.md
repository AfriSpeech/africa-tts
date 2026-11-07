# Benchmarks

## Speed

Measured with [`benchmarks/speed.py`](../benchmarks/speed.py) on a dedicated 4-vCPU Modal CPU
container ([`benchmarks/speed_modal.py`](../benchmarks/speed_modal.py)): 8 sentences in Twi, Hausa,
Yoruba, Swahili, Zulu, Amharic, Ewe and English, each language's default voice, after a warm-up
(voice prompts encoded once and cached). The Python package runs PyTorch on one thread
(pocket-tts's default); sherpa-onnx uses int8 models on four threads.

| Runtime | × real time | Real-time factor | Time to first audio (streaming, median) |
|---|---:|---:|---:|
| sherpa-onnx int8 | **2.52** | 0.40 | — |
| Python package | **2.38** | 0.42 | **0.17 s** |

For comparison, the 24-layer base model Africa TTS was distilled from runs at 0.47× real time
guided (0.72 s to first audio) and 0.82-0.90× unguided.

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

How well can the speech be understood? Each system reads the same held-out sentences, and
[Omnilingual ASR](https://huggingface.co/facebook/omniASR-LLM-7B) (`omniASR_LLM_7B_v2`)
transcribes the audio. We report the **character error rate (CER, %, lower is better)** after
NFC normalisation, lower-casing and punctuation removal.

| Languages | Real speech | **Africa TTS** (110M) | MMS-TTS (36M per language) | OmniVoice (813M) | Sunbird Orpheus (3.3B) |
|---|---:|---:|---:|---:|---:|
| 27 that MMS-TTS supports |  | **9.0** | 10.3 |  |  |
| 29 that OmniVoice supports |  | 8.8 |  | **8.4** |  |
| 11 that Sunbird Orpheus supports |  | **5.9** |  |  | 10.3 |
| 24 that have real recordings | 7.5 | 7.7 |  |  |  |

Means over the languages each comparison shares; best synthetic system in bold.

- Africa TTS is more intelligible than MMS-TTS in 21 of 27 languages and than Sunbird Orpheus in
  8 of 11.
- OmniVoice is the most intelligible system: it is better in most of its languages, by small
  margins (0.4 points on average), at about 7× Africa TTS's size.
- Distillation costs nothing: over all 45 languages Africa TTS averages 9.1%, the same as the
  guided 24-layer base model it was distilled from (unguided: 10.8%).
- With diacritics removed before scoring, all CERs drop but the ranking is the same.

Per-language results (including the base model) and the diacritic-free table:
[benchmarks/results/intelligibility.md](../benchmarks/results/intelligibility.md)
(raw: [intelligibility.json](../benchmarks/results/intelligibility.json)).

**Setup.**
- **Test sentences:** up to 50 per language for 47 trained languages (2,154 sentences). They
  come from the Omnilingual ASR corpus (test split, else dev, else train; five languages with
  tiny splits are topped up from the next split), FLEURS test, WAXAL test, or FLORES+ devtest (text only), in that order of preference.
  Sentences with digits, under 20 or over 220 characters are dropped. None are in our training
  data.
- **Voices:** Africa TTS (and its base model) uses each language's default voice at temperature 0.3. OmniVoice clones
  the same voice clip, with the clip's transcript taken from Omnilingual ASR where no full human
  transcript exists. MMS-TTS (uroman for non-Latin scripts) and Sunbird Orpheus (one fixed speaker
  per language, its recommended sampling) use their own voices.
- **Averages:** languages with fewer than 10 sentences are left out of the means.
- **Parameter counts** are measured on the released checkpoints, codec or vocoder included.

**Caveats.**
- The metric measures intelligibility only, not naturalness or speaker similarity, and it
  inherits the ASR's own errors.
- Some Omnilingual ASR corpus sentences come from its train split, which the ASR may have seen.
  This favours all systems equally.
- FLEURS sentences are FLORES sentences read aloud.
- Chitumbuka, Swati and Kimbundu have no ASR language code and are transcribed without one.
- Synthetic speech can score below real speech: many recordings are noisy or spontaneous.

**Reproduce** (GPU; scripts in [benchmarks/](../benchmarks/)):

```bash
python benchmarks/build_testset.py BENCH                      # sentences + real recordings
python benchmarks/synth_africa_tts.py BENCH --system turbo    # Africa TTS (guided, unguided: base)
python benchmarks/synth_baselines.py BENCH VOICES --system mms   # also omnivoice, orpheus
python benchmarks/transcribe.py BENCH                          # Omnilingual ASR
python benchmarks/score.py BENCH --min-n 10                    # results.json / results.md
```
