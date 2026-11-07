# Recipe: Africa TTS (92 African languages + English, IPA input)

How [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts) was made, end to end: a 24-layer
base model ([africa-tts-base](https://huggingface.co/AfriSpeech/africa-tts-base)) trained on IPA,
then distilled into the 6-layer released model.
The same steps work for adding languages: anything with an africa-g2p IPA table and aligned
speech can join.

Paths: scripts read `AFRICA_TTS_DATA` (data root, default `<repo>/data`) and `AFRICA_G2P_DIR`
(an [africa-g2p](https://github.com/AfriSpeech/africa-g2p) checkout at commit `76b53a8` with
`native/` built and its tables exported, default `<repo>/third_party/africa-g2p`); see
[paths.py](paths.py).

```bash
git clone https://github.com/AfriSpeech/africa-g2p third_party/africa-g2p
git -C third_party/africa-g2p checkout 76b53a8
cmake -S third_party/africa-g2p/native -B third_party/africa-g2p/native/build && cmake --build third_party/africa-g2p/native/build -j
python third_party/africa-g2p/native/tools/export_tables.py --all
pip install -e . -r recipes/africa/requirements.txt   # + espeak-ng 1.51 (apt install espeak-ng on Ubuntu 22.04/24.04)
```

## 1. Data (number-free, real speech only)

| Source | Script | Hours used |
|---|---|---:|
| JW recordings from [AfriSpeech Selector](https://github.com/AfriSpeech/afrispeech-selector), `afrispeech` source only, train + validation splits | `data/pull_afrispeech.py` | 383 |
| [AfriSpeech/africa-female-speech-v2](https://huggingface.co/datasets/AfriSpeech/africa-female-speech-v2) | `data/prepare_speech_parquet.py` | 192 |
| Ghanaian English (ghana-english-tts-clean2) | `data/prepare_speech_parquet.py` | 156 |
| [HiFiTTS-2](https://huggingface.co/datasets/nvidia/hifitts-2), 200 h slice | `training/scripts/prepare_data.py --hours 200` | 166 |

Every clip whose transcript contains a number (any digit, or another numeral such as Ge'ez ፩)
is dropped, and clips must be 1-30 s. Each corpus keeps its own held-out split.

## 2. Manifest, alignment, IPA

```bash
python recipes/africa/data/build_africa_manifest.py $AFRICA_TTS_DATA/africa-tts
#   keeps languages with an africa-g2p IPA table; tags rows with g2p (IPA table) and align codes
python recipes/africa/data/align_ctc.py $AFRICA_TTS_DATA/africa-tts/valid_all.jsonl $AFRICA_TTS_DATA/africa-tts/aligned_ctc/valid_all.jsonl
python recipes/africa/data/align_ctc.py $AFRICA_TTS_DATA/africa-tts/train_all.jsonl $AFRICA_TTS_DATA/africa-tts/aligned_ctc/train_all.jsonl
#   ctc-forced-aligner (MahmoudAshraf/mms-300m-1130-forced-aligner, uroman) for every language;
#   the package's C++ forced_align is swapped for torchaudio's (the C++ one corrupts memory on
#   some clips; results verified identical). 99.95% of clips aligned.
python recipes/africa/data/to_ipa.py aligned_ctc/train_all.jsonl ipa/train_all.jsonl --report ipa/report_train.json
python recipes/africa/data/to_ipa.py aligned_ctc/valid_all.jsonl ipa/valid_all.jsonl
#   word by word: africa-g2p C tables (African), espeak-ng en-us + normalisation (English)
cat ipa/train_all.jsonl ipa/hifitts2_train.jsonl > train_final.jsonl
cat ipa/valid_all.jsonl ipa/hifitts2_valid.jsonl > valid_final.jsonl
```

Alignment runs on the original spelling; the words are converted to IPA afterwards, so the
voice-prompt cut points (word boundaries) are unaffected by the G2P.

## 3. Tokenizer

```bash
python recipes/africa/data/build_ipa_tokenizer.py tok --train ipa/train_all.jsonl --valid ipa/valid_all.jsonl --vocab 4000
```

SentencePiece Unigram, 4,000 pieces, character coverage 1.0, byte fallback; corpus sampled per
language with weight ∝ √hours. 2.41 tokens per word on validation (8k: 2.21, characters: 5.86).
The result is in [tokenizer/](tokenizer/).

## 4. Base model (24 layers)

```bash
python -m training.train training/configs/finetune_africa_ipa.yaml
```

Kyutai's new-language recipe: warm start from `english_2026-04_24l`, fresh text table for the IPA
tokenizer, nothing frozen, lr 2e-4 constant after 1k warmup, batch 64, text/voice dropout 0.2.
Validation every 1k steps, keep-best on validation loss, early stopping after 20 validations
without improvement. One H200: early-stopped at 113k steps (best validation loss at 93k); the
release uses **step 100k**, chosen by listening.

## 5. Africa TTS (6 layers, distilled)

```bash
python -m training.train training/configs/depth_distill_africa_ipa.yaml
```

Kyutai's `depth_distill`: student layers seeded from teacher layers 0, 1, 2, 21, 22, 23; text
table, flow head and EOS head copied; trained to match the teacher's guided (cfg 2.0) output, so
guidance is baked in. lr 4e-4 cosine over a 200k-step ceiling (Kyutai's setting: intelligibility
reaches the teacher's by ~50k steps, speaker similarity and naturalness keep improving to
~150k), validation every 2.5k steps, keep-best, and early stopping after 8 validations (20k steps)
without improvement, so the run ends at convergence rather than at a fixed step.

The released run started with a 60k ceiling and was extended to 200k at step 30k, while
validation loss was still falling (0.0265 at 2.5k, 0.0157 at 20k, 0.0147 at 27.5k). Extending is
lossless: the learning rate is a function of the step and `max_steps`
(`training/train_utils.py:lr_at`), and checkpoints carry the optimizer state, so raising
`max_steps` and relaunching resumes from the latest checkpoint (the learning rate moves up to the
new schedule's value, 2.0e-4 -> 3.8e-4, below the 4e-4 peak).

## 6. Release

```bash
python recipes/africa/release/export_release.py training/configs/finetune_africa_ipa.yaml \
    runs/africa_ipa_teacher/checkpoint_00100000.pt release/africa-tts-base AfriSpeech/africa-tts-base
python recipes/africa/release/select_voices.py train_final.jsonl release/voices     # default voices (JW)
python recipes/africa/release/select_female_voices.py release/voices                # female voices, 17 languages
python export/onnx/export_onnx.py --model-dir release/africa-tts --out-dir onnx/africa-tts --quantize
python mobile/make_bundle.py --onnx-dir onnx/africa-tts --g2p-dir g2p --voices-dir release/voices \
    --espeak-dir espeak-en --out bundle/africa-tts
python recipes/africa/release/synth_udhr.py ...   # UDHR texts per language (Space)
python recipes/africa/release/synth_samples.py ... # Space audio, made with the africa_tts package
python recipes/africa/release/make_space.py ... ; python recipes/africa/release/make_card.py ...
python recipes/africa/release/make_video.py space_samples out.mp4 --label africa-tts [--langs twi,ewe,...]  # slideshow
python recipes/africa/release/experimental_languages.py ipa/train_all.jsonl africa_tts/languages_experimental.json --md table.md
```

`synth_demo.py` / `synth_ood.py` are the listening tests used to pick the checkpoint (held-out
JW sentences, FLORES+ devtest sentences).
