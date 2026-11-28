# Language-tag recipe template

This template shows how to add a small set of explicit language/style tokens to an Africa TTS
teacher and then distill it into a Pocket TTS student. It is generic; replace the example tokenizer,
manifest paths, tags, and teacher checkpoint with your own.

The important contract is that the tag is part of the text seen by the model. The manifest stores it
in `condition_prefix`; the dataloader prepends it at tokenization time; inference prepends the same
token before calling Pocket TTS. If any of those three disagree, the model will not follow the tag.

## 1. Enable `condition_prefix`

Apply `patches/condition_prefix.patch` from the repository root:

```bash
git apply training/recipes/lang_tags/patches/condition_prefix.patch
```

It adds a `condition_prefix` field to `Entry` and prepends it after final-punctuation dropout in
all three tokenization paths (raw audio, cut audio, and cached latents). Keeping the prefix inside
`_tokens` means word cuts apply to the transcript only.

## 2. Build manifests

Input rows are ordinary Africa TTS manifest rows with at least `path`, `duration`, and `transcript`.
The script copies every field and sets `condition_prefix` from a tag field:

```bash
python training/recipes/lang_tags/scripts/build_manifests.py \
  --inputs data/raw/train.jsonl \
  --out data/lang_tags/train.jsonl \
  --tag-field language \
  --tags eng tw \
  --tag-prefix '<lang:{tag}>'
```

Do the same for `valid.jsonl`. Tokens such as `<lang:eng>` and `<lang:tw>` must be the exact strings
reserved in the tokenizer.

## 3. Build the tokenizer

Reserve the tags as SentencePiece user-defined symbols:

```bash
PY=python TAGS=(
  '<lang:eng>' '<lang:tw>'
)
TOK=training/recipes/lang_tags/tokenizer

bash training/recipes/lang_tags/scripts/build_tokenizer.sh \
  "$TOK" data/lang_tags/train.jsonl data/lang_tags/valid.jsonl "${TAGS[@]}"
```

Use the converted `ipa_sp4000.json` and its vocabulary size in every training and inference config.
A fresh tokenizer with new reserved symbols requires `reset_text_embedding: true` for the teacher.

## 4. Train the teacher

```bash
python -m training.train training/recipes/lang_tags/configs/finetune_teacher.yaml
```

The teacher is a 24-layer finetune from the English base. It samples with `cfg=2.0`; the student
will have that guidance baked into its weights.

## 5. Distill

Same-width depth distillation is the default and needs no PCA seed because the text table, flow
head, EOS head, and latent stats copy directly from the teacher:

```bash
python -m training.train training/recipes/lang_tags/configs/distill_same_width.yaml
```

For a narrower student, build a PCA/shrink seed first. This projects the teacher's text table,
including the learned tag rows, into the student width. Without it those rows are randomly
initialized, which is a common cause of students that ignore the tags.

```bash
TOKENIZER=training/recipes/lang_tags/tokenizer/ipa_sp4000.json \
LAYERS=8 D_MODEL=512 NUM_HEADS=8 HIDDEN_SCALE=1 LOOKUP_DIM=512 Z_DIM=1024 \
OUT_SEED=runs/lang_tags_init_pca.pt \
bash training/recipes/lang_tags/scripts/make_pca_seed.sh \
  runs/lang_tags_teacher/args.yaml \
  runs/lang_tags_teacher/checkpoint_00200000.pt \
  runs/lang_tags_init_pca.pt

python -m training.train training/recipes/lang_tags/configs/distill_narrow.yaml
```

Keep `flow_lm.z_dim` equal to the teacher's z width so the frozen heads stay compatible. Use
`text_dropout: 0.0`, `voice_dropout: 0.0`, `distill_cfg_coef: 2.0`, and `sample_cfg_coef: 1.0`.

## 6. Optional SFT for very small students

Kyutai used an extra supervised fine-tune after the 18M narrow distillation. It is not needed for
the same-width distilled model, and it can degrade the guidance baked in by direct distillation.
Use it only if CER/text-following on a narrow student is poor after distillation:

```bash
python -m training.train training/recipes/lang_tags/configs/finetune_small_student.yaml
```

## 7. Inference

Distilled students are sampled with `cfg=1.0`, with no CFG wrapper. The tag must be prepended:

```bash
python training/recipes/lang_tags/scripts/build_inference_config.py \
  --run-dir runs/lang_tags_distill_6l \
  --checkpoint runs/lang_tags_distill_6l/best_model.safetensors \
  --tokenizer training/recipes/lang_tags/tokenizer/ipa_sp4000.json \
  --out runs/lang_tags_distill_6l/inference_config.yaml

python training/recipes/lang_tags/scripts/infer.py \
  --config runs/lang_tags_distill_6l/inference_config.yaml \
  --checkpoint runs/lang_tags_distill_6l/best_model.safetensors \
  --tokenizer training/recipes/lang_tags/tokenizer/ipa_sp4000.json \
  --voice data/voices/twi.wav \
  --tag tw \
  --text "ɛna yɛn nhyiam no bɛkɔ so wɔ anɔpa." \
  --out out_tw.wav
```

For the teacher, use `recipes/africa/guidance.py` with `cfg=2.0` and the same prefix.
