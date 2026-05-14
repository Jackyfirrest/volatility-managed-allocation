# IEOR 198 Final Project Summary

## Project
Enhanced Volatility-Managed Allocation with GARCH, CVaR, and a Gold Defensive Sleeve

This project extends the original SPY/TLT proposal by adding GLD as a third asset. The upgraded
strategy uses monthly walk-forward rebalancing, GARCH(1,1) volatility forecasts, and CVaR-aware
portfolio construction to dynamically allocate across SPY, TLT, and GLD.

## Why the enhancement matters
- The original SPY/TLT strategy struggled in the 2022-2024 rate-hike regime because stocks and long-duration bonds both sold off.
- Adding GLD introduces a more regime-resilient defensive sleeve.
- The enhanced strategy improves Sharpe ratio and reduces drawdown relative to both the original dynamic model and the classic 60/40 benchmark.

## Average allocation
- SPY: 39.7%
- TLT: 28.0%
- GLD: 32.3%

## Full-Sample Metrics
| Strategy | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown | Monthly CVaR 95% |
| --- | --- | --- | --- | --- | --- | --- |
| Original Dynamic SPY/TLT | 2.1825 | 0.0862 | 0.0970 | 0.8891 | -0.2827 | 0.0570 |
| Enhanced Dynamic SPY/TLT/GLD | 2.3750 | 0.0908 | 0.0916 | 0.9913 | -0.2267 | 0.0463 |
| Static 60/40 | 2.5348 | 0.0944 | 0.1005 | 0.9392 | -0.2623 | 0.0599 |
| Equal Weight 1/3 | 1.6332 | 0.0716 | 0.0945 | 0.7581 | -0.2124 | 0.0492 |
| Buy and Hold SPY | 5.0217 | 0.1368 | 0.1424 | 0.9607 | -0.2393 | 0.0841 |

## Enhanced Strategy Regime Breakdown
| Regime | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown | Monthly CVaR 95% |
| --- | --- | --- | --- | --- | --- | --- |
| GFC Recovery to Pre-COVID | 1.5442 | 0.1093 | 0.0776 | 1.4097 | -0.0793 | 0.0332 |
| COVID Shock | 0.1130 | 0.1130 | 0.1017 | 1.1121 | -0.0731 | 0.0224 |
| Rate Hike Regime | 0.1339 | 0.0428 | 0.1221 | 0.3504 | -0.1884 | 0.0517 |

## Deliverables
- `outputs/equity_curves.png`
- `outputs/enhanced_dynamic_weights.png`
- `outputs/enhanced_forecast_volatility.png`
- `outputs/performance_metrics.csv`
- `outputs/crisis_metrics.csv`
- `final_report.md`
- `final_report.tex`
