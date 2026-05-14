# IEOR 198 Quantitative Finance Final Project

This repository contains a complete quantitative research project for `IEOR 198: Introduction to Quantitative Finance` at UC Berkeley.

## Project summary
The project starts from a volatility-managed `SPY/TLT` allocation idea and improves it into an enhanced `SPY/TLT/GLD` strategy. The final model combines:
- GARCH(1,1) volatility forecasting
- CVaR-based tail-risk control
- turnover-aware dynamic rebalancing
- a gold defensive sleeve to improve robustness in regimes where stocks and long-duration bonds fall together

The main result is that the enhanced dynamic strategy improves risk-adjusted performance relative to the original dynamic model and also beats a classic `60/40` portfolio on Sharpe ratio.

The repository also includes a PPO-based DRL extension. In a strict 2021-2024 out-of-sample test, the DRL policy generates stronger return and Sharpe ratio than the static baselines and the rule-based allocators, but it still exhibits materially worse tail risk than the enhanced GARCH-CVaR portfolio.

## Repository structure
- `run_final_project.py`: main backtest and output generation script
- `run_drl_extension.py`: PPO-based DRL asset allocation extension
- `build_report_pdf.py`: utility script that assembles a PDF report from generated figures
- `scripts/reproduce.sh`: one-command bash workflow for rebuilding the project
- `requirements.txt`: Python dependencies for reproduction
- `final_report.tex`: LaTeX version of the final paper
- `final_report.pdf`: compiled report for sharing
- `data/`: downloaded ETF price data used by the project
- `outputs/`: performance tables, dynamic weights, volatility forecasts, and figures
- `references/`: original proposal and course project PDFs

## Universe and benchmarks
Assets:
- `SPY`: U.S. equities
- `TLT`: long-duration U.S. Treasuries
- `GLD`: gold

Benchmarks:
- original dynamic `SPY/TLT`
- static `60/40`
- equal weight `1/3`
- buy-and-hold `SPY`

## Reproduce
The fastest way to reproduce the project from a Unix-like shell is:

```bash
bash scripts/reproduce.sh
```

This script:
- creates a local virtual environment in `.venv/`
- installs the required Python packages
- reruns the full baseline backtest
- trains and evaluates the PPO-based DRL extension
- regenerates the companion PDF report

## Manual run
```bash
python -m pip install -r requirements.txt
python run_final_project.py
python run_drl_extension.py
python build_report_pdf.py
```

## Main deliverables
- `final_report.tex`
- `final_report.pdf`
- `outputs/performance_metrics.csv`
- `outputs/crisis_metrics.csv`
- `outputs/equity_curves.png`
- `outputs/enhanced_dynamic_weights.png`
- `outputs/enhanced_forecast_volatility.png`
- `outputs/drl_performance_metrics.csv`
- `outputs/drl_vs_baseline_metrics.csv`
- `outputs/drl_equity_curves.png`
