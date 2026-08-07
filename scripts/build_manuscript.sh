#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
TEX_ENGINE="${TECTONIC:-tectonic}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/fourier-circuit-grokking-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/fourier-circuit-grokking-cache}"

cd "$ROOT"
"$PYTHON" scripts/generate_figures.py
"$PYTHON" scripts/generate_tables.py
"$PYTHON" scripts/check_headline_numbers.py
"$PYTHON" scripts/write_compact_checksums.py
"$PYTHON" scripts/lint_manuscript.py

rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.run.xml main.synctex.gz
for pass in 1 2 3; do
  echo "Tectonic pass $pass/3"
  "$TEX_ENGINE" --keep-logs --keep-intermediates main.tex
done

"$PYTHON" scripts/check_latex_log.py main.log --blg main.blg
echo "Built $ROOT/main.pdf"
