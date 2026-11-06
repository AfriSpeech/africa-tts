# Twi recipe: a new language on the English teacher, with a merged tokenizer

How the Twi Pocket TTS was built (Ghana NLP, November 2026), written so the next
language can reuse it. Scripts assume they run from a project root holding `data/`
(on the H200: `/mnt/volume_d2wey28/projects/pocket-tts-twi`).

## Pipeline

1. **Data** — `ghanaopenai/ghana-podcasts-twi-en` podcast chunks, extracted to one 24 kHz
   FLAC per episode with manifest windows (`scripts/download_dataset.py`,
   `scripts/prepare_twi.py`).
2. **One voice** — every episode voice-converted to a single female voice with
   `AfriSpeech/africa-female-vc` (`clear-high-slow`, 50 steps), resampled to 24 kHz
   (`scripts/convert_dataset_vc.py`; uploads + cleanup in `scripts/auto_sync_hf.py`).
   We stopped at ~350 h of Twi, plenty for a single voice.
   *Why VC:* raw podcast audio scores UTMOS ~1.5 (radio compression, music beds, noise)
   vs ~3.0 after conversion (`scripts/score_twi.py` measures UTMOS + prompt/target
   speaker consistency, if you want to try raw multi-speaker data instead).
3. **Alignment** — Meta MMS 1B CTC forced alignment, `--model facebook/mms-1b-all
   --lang aka` (`training/scripts/align_data.py` gained `--lang` for MMS adapters).
   Alignment runs on the real orthography: no character mapping anywhere.
4. **Tokenizer** — English tokenizer merged with Twi pieces (below).
5. **Teacher** — `configs/finetune_twi.yaml`: warm start from the 24-layer English teacher,
   lr 2e-4 (new-language rate), 200k steps, last 20 checkpoints + best-by-validation
   (`keep_best_metric`).
6. **Fast model** — `configs/depth_distill_twi.yaml`: 24 → 6 layers (316M → 89M FlowLM),
   CFG baked in (sample with cfg 1); `scripts/run_distill_twi.sh [--wait]`.

`scripts/finish_pipeline_vc.sh` runs 3 → 5 end to end.

## The tokenizer: merge, don't replace

Options we measured on Twi validation transcripts (tokens per word):

| Tokenizer | Twi | English | Text table |
|---|---|---|---|
| Released English (byte fallback for ɛ/ɔ) | 2.97 | 1.79 | pretrained |
| Twi-only SentencePiece (4k) | 1.33 | — | **reset** (random) |
| Character mapping ɛ→q, ɔ→x | 2.58 | 1.79 | pretrained, but wrong sounds + collides with English x/q |
| **English + Twi pieces, English-protected (used)** | **1.62** | **1.79, 0.0% lines changed** | **pretrained + 1,921 new rows** |

Steps (`tokenizer/` holds the outputs):

```bash
# 1. Twi SentencePiece, same spec as the released tokenizers (unigram, identity norm, ...)
#    on a large clean Twi corpus (ghanaopenai/pristine-twi, 21M sentences, cased) + transcripts.
python data/build_tok_corpus.py            # scripts/build_tok_corpus.py
uv run python -m training.scripts.train_tokenizer data/tok/twi_sp4000 data/tok/twi_corpus.txt --vocab-size 4000
uv run python -m training.scripts.convert_tokenizer data/tok/twi_sp4000.model --out data/tok/twi_sp4000.json

# 2. Merge into the English tokenizer. Every English id is kept; Twi pieces that would change
#    how English text tokenizes are left out (pruned against ~310k English lines, verified
#    on held-out English); each new row starts as the mean of the English rows it replaces.
python data/build_protect_en.py            # scripts/build_protect_en.py
uv run python -m training.scripts.merge_tokenizers \
    hf://kyutai/pocket-tts-without-voice-cloning/languages/english_2026-04_24l/tokenizer.json@00eac05ed3d16bdc3f6b5d598874019c34a89214 \
    data/tok/twi_sp4000.json data/tok/en_twi_4000.json \
    --init-char ɛ=e --init-char ɔ=o --init-char Ɛ=E --init-char Ɔ=O \
    --protect-base data/tok/protect_en.txt \
    --eval-base data/tok/eval_en.txt --eval-add data/tok/eval_twi.txt
```

Then in the training config:

```yaml
model_overrides:
  flow_lm.lookup_table.tokenizer_path: data/tok/en_twi_4000.json
  flow_lm.lookup_table.n_bins: 5921          # the merged vocab size
reset_text_embedding: false
text_embedding_init_file: data/tok/en_twi_4000.json.init.json
```

`text_embedding_init_file` / `text_embedding_init` (training/args.py) grow the pretrained
table at load time (`grow_text_table` in training/modules/builders.py): English rows keep
their ids, new rows start from the given English rows' mean, padding moves to the end.

**Vocab size.** A token only learns its sound from the audio it appears in. With 4k Twi
pieces, 882 of 1,921 new tokens get ≥100 uses in ~350 h of transcripts (median 73); with
8k pieces Twi is only 4% shorter (1.56) but 1,765 new tokens get <20 uses (median 25).
Size the added vocabulary to the audio you have, not to the text corpus.

## Applying this to another language

1. Get a large clean text corpus in the language's real orthography.
2. Train a ~4k SentencePiece on it (`train_tokenizer`), convert to json.
3. `merge_tokenizers` with `--protect-base` English text, `--init-char` for letters
   English only covers through byte fallback (map to the nearest-sounding letter), and
   eval files for both languages; check 0.0% English lines changed.
4. Count how often new pieces occur in your training transcripts before choosing a size.
5. Set `n_bins`, `tokenizer_path`, `text_embedding_init_file` in the teacher and distill
   configs (the distill student copies the teacher's table).

The inference config for the result needs the same `tokenizer_path` and `n_bins`.
