# Contributing to Africa TTS

Africa TTS is an ongoing, community project: the goal is good, free text-to-speech for every
African language, on every device. The first release covers 92 African languages and English;
many are still thin (under 2 hours of speech), many languages are missing, and plenty of the
pipeline can be improved. Contributions of every size are welcome, from a native speaker telling
us a word sounds wrong to a new language with its own recordings.

## Ways to help

| You have... | Help with | Where |
|---|---|---|
| A language the model was not trained on | Try it (657 languages have an IPA table) and report how it sounds | [docs/EXPERIMENTAL_LANGUAGES.md](docs/EXPERIMENTAL_LANGUAGES.md) |
| A good ear for a language | Listen to the [samples](https://huggingface.co/spaces/AfriSpeech/africa-tts-samples) and report what is wrong: tone, vowels, stress, rhythm, words read like another language | [issue: pronunciation](../../issues/new?template=pronunciation.yml) |
| Knowledge of a writing system | Fix or extend the language's africa-g2p table (tone marks, missing letters, digraphs) | [AfriSpeech/africa-g2p](https://github.com/AfriSpeech/africa-g2p) |
| Speech recordings with transcripts | Add or grow a language (see "Adding a language" below) | [issue: new language / data](../../issues/new?template=new-language.yml) |
| Evaluation skills | Per-language intelligibility (ASR CER), speaker similarity, MOS listening tests | `recipes/africa/release/`, new `eval/` scripts |
| Mobile / C experience | Android NDK build, an example app, iOS, espeak-ng 1.51 builds | `mobile/` |
| ML experience | Training improvements, distillation, number reading, streaming | `training/`, `recipes/africa/` |
| Time to write | Docs, tutorials, translations of the README | anywhere |

Good first issues are labelled [`good first issue`](../../labels/good%20first%20issue).

## The one rule: the front end must match training

The model only knows the IPA it was trained on. If text is converted differently at inference,
quality drops silently. So:

- **A change to an africa-g2p table is a model change.** It improves the IPA, but the released
  models were trained on the old IPA. Table fixes are very welcome; they go into the next
  training run, and the pinned africa-g2p commit (`pyproject.toml`, `mobile/frontend/CMakeLists.txt`,
  the model cards) moves only together with a retrained model.
- **The Python and C front ends must stay byte-identical** to the training IPA. If you touch
  `africa_tts/frontend.py` or `mobile/frontend/`, run the parity check (below) and include its
  output in the pull request.
- **English uses espeak-ng 1.51.** 1.52 changes about 2% of English words.

## Development setup

```bash
git clone https://github.com/AfriSpeech/africa-tts && cd africa-tts
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,export]"
sudo apt install espeak-ng          # 1.51 on Ubuntu 22.04/24.04; brew install espeak-ng on macOS
```

Checks before a pull request:

```bash
ruff check . && ruff format --check .                     # style: line length 100
pytest training/tests                                     # training code
# front end (if touched):
cmake -S mobile/frontend -B build/frontend && cmake --build build/frontend -j
huggingface-cli download AfriSpeech/africa-tts --include "g2p/*" --local-dir hf
python mobile/frontend/tests/check_parity.py build/frontend/att_ipa hf/g2p
# ONNX export (if touched): export/onnx/test_parity.py, test_tokenizer.py, test_sherpa.py
```

## Adding a language

A language can join the next training run when it has:

1. **An africa-g2p IPA table.** Check with `python -c "from africa_g2p import g2p; print(g2p('...', '<iso>', output='ipa'))"`.
   If the language has no table, or the table is wrong, start in
   [africa-g2p](https://github.com/AfriSpeech/africa-g2p).
2. **Speech with transcripts**, ideally at least 2 hours (10+ is much better), clips of 1-30 s,
   read by speakers who agreed to the use. Transcripts must be in the standard orthography.
   Clips whose transcripts contain digits are dropped, so write numbers as words.
3. **A clear license** that allows training and releasing a model. Tell us the source and terms
   in the issue; we cannot use data whose rights are unclear.

Then: prepare manifests like `recipes/africa/data/prepare_speech_parquet.py`, run the alignment
and IPA steps in [recipes/africa/README.md](recipes/africa/README.md), and open a pull request
with the per-language report (`to_ipa.py --report`: clips, unconverted letters) and a few
listening samples. We batch new languages into training runs and credit contributors in the
model card.

## Reporting pronunciation problems

The most useful report has: the language code, the exact text, what you heard, what it should
be. `africa-tts --ipa-only "<text>" --lang <code>` shows the
IPA the model read: if the IPA is already wrong, the fix belongs in africa-g2p; if the IPA is
right but the audio is wrong, it is a model issue (data or training).

## Pull requests

- One topic per pull request; describe what changed and how you tested it.
- Keep the training code compatible with upstream [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
  where possible: changes that are useful beyond Africa TTS can go upstream too.
- Scripts in `recipes/africa/` record how the released models were made. Fix bugs freely; for a
  different method, add a new script or option rather than silently changing what a release used.
- Never commit tokens, private data, or audio you do not have the rights to share.

## Licensing

Code contributions are accepted under the MIT License (see [LICENSE](LICENSE)). Model weights and
default voices are CC-BY-NC-4.0. Data you contribute keeps its own license, which must allow
training and releasing models; tell us what it is.

## Conduct

Be kind and assume good faith. This project serves many languages and communities: respect
native speakers' knowledge of their own languages, and credit people's work. Report problems to
the maintainers via a private message to [AfriSpeech on Hugging Face](https://huggingface.co/AfriSpeech)
or an issue.
