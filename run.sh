#!/usr/bin/env bash
# One-shot: build corpus -> questions -> benchmark -> report.  Usage: ./run.sh [small|medium|large]
set -euo pipefail
SCALE="${1:-}"
[ -n "$SCALE" ] && sed -i.bak "s/^scale: .*/scale: $SCALE/" config.yaml
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY first}"
python tests/test_offline.py
python -m bench.build_corpus
python -m bench.qa_gen
python -m bench.run_benchmark all
python -m bench.report
echo "Open results/report.html"
