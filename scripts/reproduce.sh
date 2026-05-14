#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python was not found in PATH." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python run_final_project.py
python run_drl_extension.py
python report/build_report_pdf.py

echo
echo "Reproduction complete."
echo "Main outputs:"
echo "  - outputs/performance_metrics.csv"
echo "  - outputs/crisis_metrics.csv"
echo "  - outputs/equity_curves.png"
echo "  - outputs/enhanced_dynamic_weights.png"
echo "  - outputs/enhanced_forecast_volatility.png"
echo "  - outputs/drl_performance_metrics.csv"
echo "  - outputs/drl_equity_curves.png"
echo "  - report/final_report.pdf"
