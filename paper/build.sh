#!/bin/bash
# Build the arXiv paper: tables from the benchmark results, one flat main.tex (tables inlined),
# compiled with the texlive Docker image; writes arxiv/{main.tex,main.bbl,africa-tts.pdf,
# arxiv-submission.tar.gz}.
# Usage: paper/build.sh RESULTS.json SPEED.json
set -e
P=$(cd "$(dirname "$0")" && pwd)
python3 "$P/make_tables.py" "$1" "$2" "$P/tables" > /dev/null
python3 - "$P" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
src = (p / "main.tex").read_text()
flat = re.sub(r"\\input\{tables/(\w+)\}", lambda m: (p / "tables" / f"{m.group(1)}.tex").read_text().rstrip("\n"), src)
(p / "arxiv" / "main.tex").write_text(flat)
PY
cd "$P/arxiv"
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/w -w /w texlive/texlive:latest-medium sh -c \
  "pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main | grep -E 'Warning|Output written' || true"
mv main.pdf africa-tts.pdf
rm -f main.aux main.blg main.log main.out
tar czf arxiv-submission.tar.gz main.tex main.bbl refs.bib
echo "built $P/arxiv/africa-tts.pdf"
