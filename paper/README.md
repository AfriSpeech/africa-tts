# Paper

*Africa TTS: Fast, CPU-Friendly Speech Synthesis for 92 African Languages from Phonemes.*
Every number in its tables comes from the benchmark outputs in
[`benchmarks/results/`](../benchmarks/results/), through `make_tables.py`.

```bash
paper/build.sh benchmarks/results/intelligibility.json benchmarks/results/speed_modal_4vcpu.json
```

This writes the tables and headline numbers (`paper/tables/`), a flat `arxiv/main.tex` with the
tables inlined, and compiles it with the `texlive/texlive:latest-medium` Docker image into
`arxiv/africa-tts.pdf` and `arxiv/arxiv-submission.tar.gz` (`main.tex`, `main.bbl`, `refs.bib`;
the `.bbl` is included because arXiv does not run BibTeX reliably). Edit `paper/main.tex`, not
`arxiv/main.tex`.
