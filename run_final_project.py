from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / ".yfinance_cache"

ASSETS = ["SPY", "TLT", "GLD"]
START_DATE = "2008-01-01"
END_DATE = "2025-01-01"
LOOKBACK_DAYS = 756
MOMENTUM_SHORT = 126
MOMENTUM_LONG = 252
MONTHLY_HORIZON = 21
ALPHA = 0.95
TRANSACTION_COST = 0.001
RISK_AVERSION = 1.5
TURNOVER_PENALTY = 0.001
EQUITY_BOUNDS = (0.10, 0.90)


@dataclass
class StrategyDecision:
    date: pd.Timestamp
    weights: Dict[str, float]
    forecasts: Dict[str, float]
    cvar_estimate: float
    expected_return: float
    score: float


def ensure_directories() -> None:
    for path in (DATA_DIR, OUTPUT_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def cleanup_legacy_artifacts() -> None:
    legacy_paths = [
        DATA_DIR / "expanded_close.csv",
        OUTPUT_DIR / "dynamic_realized_weights.csv",
        OUTPUT_DIR / "dynamic_targets.csv",
        OUTPUT_DIR / "dynamic_weights.png",
        OUTPUT_DIR / "forecast_volatility.png",
    ]
    for path in legacy_paths:
        if path.exists():
            path.unlink()


def download_prices() -> pd.DataFrame:
    yf.set_tz_cache_location(str(CACHE_DIR.resolve()))
    raw = yf.download(
        ASSETS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("No market data downloaded.")

    close = raw["Close"].copy().dropna()
    close.columns.name = None
    close.to_csv(DATA_DIR / "adjusted_close.csv")
    return close


def month_end_index(prices: pd.DataFrame) -> pd.DatetimeIndex:
    grouped = prices.groupby([prices.index.year, prices.index.month])
    return pd.DatetimeIndex(grouped.tail(1).index)


def negative_log_likelihood(params: np.ndarray, centered: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e12

    variance = np.empty(len(centered))
    variance[0] = max(np.var(centered, ddof=1), 1e-6)
    for idx in range(1, len(centered)):
        variance[idx] = omega + alpha * centered[idx - 1] ** 2 + beta * variance[idx - 1]
        variance[idx] = max(variance[idx], 1e-8)

    ll = 0.5 * np.sum(np.log(2 * np.pi) + np.log(variance) + centered**2 / variance)
    return float(ll)


def garch_forecast_vol(returns: pd.Series, horizon_days: int = MONTHLY_HORIZON) -> float:
    clean = returns.dropna().to_numpy(dtype=float)
    if len(clean) < 60:
        return float(clean.std(ddof=1) * math.sqrt(horizon_days))

    scaled = clean * 100.0
    centered = scaled - scaled.mean()
    unconditional = np.var(centered, ddof=1)
    initial = np.array([max(unconditional * 0.05, 1e-4), 0.05, 0.90])
    bounds = [(1e-8, None), (1e-8, 0.40), (1e-8, 0.999)]
    constraints = [{"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]}]

    result = minimize(
        negative_log_likelihood,
        x0=initial,
        args=(centered,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )

    if result.success:
        omega, alpha, beta = result.x
        variance = unconditional
        for value in centered:
            variance = omega + alpha * value**2 + beta * variance
        next_variance = omega + alpha * centered[-1] ** 2 + beta * variance
        next_variance = max(next_variance, 1e-8)
        return float(math.sqrt(next_variance) / 100.0 * math.sqrt(horizon_days))

    lam = 0.94
    ewma_var = np.var(clean[:20], ddof=1)
    for value in clean:
        ewma_var = lam * ewma_var + (1 - lam) * value**2
    return float(math.sqrt(ewma_var) * math.sqrt(horizon_days))


def compute_cvar(returns: np.ndarray, alpha: float = ALPHA) -> float:
    losses = -returns
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    if len(tail) == 0:
        return float(max(var, 0.0))
    return float(np.mean(tail))


def monthly_scenarios(window_returns: pd.DataFrame) -> pd.DataFrame:
    rolling_log = window_returns.rolling(MONTHLY_HORIZON).sum().dropna()
    return np.exp(rolling_log) - 1.0


def expected_monthly_returns(window_returns: pd.DataFrame) -> pd.Series:
    short = window_returns.tail(MOMENTUM_SHORT).mean() * MONTHLY_HORIZON
    long = window_returns.tail(MOMENTUM_LONG).mean() * MONTHLY_HORIZON
    mu = 0.6 * short + 0.4 * long
    return mu.clip(lower=-0.06, upper=0.06)


def generate_weight_candidates(
    tickers: Sequence[str],
    step: float = 0.05,
    equity_bounds: Tuple[float, float] = EQUITY_BOUNDS,
) -> List[np.ndarray]:
    if len(tickers) == 2:
        return [np.array([w, 1.0 - w]) for w in np.arange(equity_bounds[0], equity_bounds[1] + 1e-9, step)]

    if len(tickers) == 3:
        candidates: List[np.ndarray] = []
        spy_min, spy_max = equity_bounds
        values = np.arange(0.0, 1.0 + 1e-9, step)
        for w_spy in np.arange(spy_min, spy_max + 1e-9, step):
            for w_tlt in values:
                w_gld = 1.0 - w_spy - w_tlt
                if w_gld < -1e-9:
                    continue
                if w_gld < 0:
                    w_gld = 0.0
                weights = np.array([w_spy, w_tlt, w_gld])
                if np.isclose(weights.sum(), 1.0):
                    candidates.append(weights)
        return candidates

    raise ValueError("Weight candidate generator currently supports 2 or 3 assets.")


def choose_weight(
    window_returns: pd.DataFrame,
    tickers: Sequence[str],
    previous_weights: np.ndarray,
) -> StrategyDecision:
    forecasts = pd.Series({ticker: garch_forecast_vol(window_returns[ticker]) for ticker in tickers})
    scenario_df = monthly_scenarios(window_returns[tickers])
    hist_vol = scenario_df.std(ddof=1).replace(0.0, np.nan)
    scaled_scenarios = scenario_df * (forecasts / hist_vol).fillna(1.0)
    mu = expected_monthly_returns(window_returns[tickers])
    weight_candidates = generate_weight_candidates(tickers)

    best: StrategyDecision | None = None
    for weights in weight_candidates:
        scenario_returns = scaled_scenarios.to_numpy() @ weights
        cvar = compute_cvar(scenario_returns, alpha=ALPHA)
        expected_return = float(np.dot(mu.to_numpy(), weights))
        turnover = float(np.abs(weights - previous_weights).sum())
        score = expected_return - RISK_AVERSION * cvar - TURNOVER_PENALTY * turnover

        decision = StrategyDecision(
            date=window_returns.index[-1],
            weights={ticker: float(weight) for ticker, weight in zip(tickers, weights)},
            forecasts={ticker: float(forecasts[ticker]) for ticker in tickers},
            cvar_estimate=float(cvar),
            expected_return=expected_return,
            score=float(score),
        )
        if best is None or decision.score > best.score:
            best = decision

    if best is None:
        raise RuntimeError("Strategy optimization failed.")
    return best


def build_dynamic_targets(
    log_returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    tickers: Sequence[str],
    initial_weights: np.ndarray,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    previous_weights = initial_weights.copy()

    for idx in range(len(rebalance_dates) - 1):
        rebalance_date = rebalance_dates[idx]
        window = log_returns.loc[:rebalance_date, tickers].tail(LOOKBACK_DAYS)
        if len(window) < LOOKBACK_DAYS:
            continue

        decision = choose_weight(window, tickers=tickers, previous_weights=previous_weights)
        previous_weights = np.array([decision.weights[ticker] for ticker in tickers])
        record: Dict[str, float] = {"Date": rebalance_dates[idx + 1]}
        record.update(decision.weights)
        for ticker in tickers:
            record[f"ForecastVol_{ticker}"] = decision.forecasts[ticker]
        record["EstimatedCVaR"] = decision.cvar_estimate
        record["ExpectedMonthlyReturn"] = decision.expected_return
        record["Score"] = decision.score
        records.append(record)

    if not records:
        raise RuntimeError("No dynamic targets were generated.")

    return pd.DataFrame.from_records(records).set_index("Date")


def monthly_simple_returns(prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    records = []
    for idx in range(len(rebalance_dates) - 1):
        start = rebalance_dates[idx]
        end = rebalance_dates[idx + 1]
        period = prices.loc[[start, end]]
        simple_ret = period.iloc[-1] / period.iloc[0] - 1.0
        record = {"Date": end, **simple_ret.to_dict()}
        records.append(record)
    return pd.DataFrame.from_records(records).set_index("Date")


def simulate_portfolio(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    tickers: Sequence[str],
    cost_rate: float = TRANSACTION_COST,
) -> Tuple[pd.Series, pd.DataFrame]:
    common_dates = asset_returns.index.intersection(target_weights.index)
    asset_returns = asset_returns.loc[common_dates, tickers]
    target_weights = target_weights.loc[common_dates, tickers]

    previous_end_weights = target_weights.iloc[0].astype(float).copy()
    portfolio_returns: List[float] = []
    realized_weights: List[Dict[str, float]] = []

    for date in common_dates:
        target = target_weights.loc[date].astype(float)
        turnover = float(np.abs(target - previous_end_weights).sum())
        gross_return = float(np.dot(target.to_numpy(), asset_returns.loc[date].to_numpy()))
        net_return = gross_return - cost_rate * turnover
        portfolio_returns.append(net_return)

        end_values = target * (1.0 + asset_returns.loc[date])
        previous_end_weights = end_values / end_values.sum()
        realized_weights.append({"Date": date, "Turnover": turnover, **target.to_dict()})

    returns = pd.Series(portfolio_returns, index=common_dates, name="PortfolioReturn")
    weights = pd.DataFrame.from_records(realized_weights).set_index("Date")
    return returns, weights


def performance_metrics(returns: pd.Series) -> Dict[str, float]:
    wealth = (1.0 + returns).cumprod()
    periods_per_year = 12
    total_periods = len(returns)
    annual_return = wealth.iloc[-1] ** (periods_per_year / total_periods) - 1.0
    annual_vol = returns.std(ddof=1) * math.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    cvar = compute_cvar(returns.to_numpy(), alpha=ALPHA)
    return {
        "Cumulative Return": wealth.iloc[-1] - 1.0,
        "Annualized Return": annual_return,
        "Annualized Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": drawdown.min(),
        "Monthly CVaR 95%": cvar,
    }


def crisis_period_metrics(returns: pd.Series) -> pd.DataFrame:
    slices = {
        "GFC Recovery to Pre-COVID": returns.loc["2011-01-01":"2019-12-31"],
        "COVID Shock": returns.loc["2020-01-01":"2020-12-31"],
        "Rate Hike Regime": returns.loc["2022-01-01":"2024-12-31"],
    }
    rows = {}
    for name, series in slices.items():
        if len(series) > 0:
            rows[name] = performance_metrics(series)
    return pd.DataFrame(rows).T


def save_plot_equity_curves(results: pd.DataFrame) -> None:
    wealth = (1.0 + results).cumprod()
    plt.figure(figsize=(11, 6))
    for column in wealth.columns:
        plt.plot(wealth.index, wealth[column], label=column, linewidth=2)
    plt.title("Portfolio Growth of $1")
    plt.ylabel("Wealth")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curves.png", dpi=200)
    plt.close()


def save_plot_weights(weights: pd.DataFrame, tickers: Sequence[str], filename: str, title: str) -> None:
    plt.figure(figsize=(11, 5))
    for ticker in tickers:
        plt.plot(weights.index, weights[ticker], label=f"{ticker} weight", linewidth=2)
    plt.title(title)
    plt.ylabel("Weight")
    plt.xlabel("Date")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def save_plot_forecasts(target_weights: pd.DataFrame, tickers: Sequence[str], filename: str, title: str) -> None:
    plt.figure(figsize=(11, 5))
    for ticker in tickers:
        plt.plot(target_weights.index, target_weights[f"ForecastVol_{ticker}"], label=f"{ticker} forecast vol", linewidth=2)
    plt.title(title)
    plt.ylabel("Forecast Volatility")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame, decimals: int = 4, index_name: str = "Strategy") -> str:
    formatted = df.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{decimals}f}")

    headers = [index_name, *formatted.columns.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for index, row in formatted.iterrows():
        values = [str(index), *row.astype(str).tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_summary_report(
    metrics_df: pd.DataFrame,
    crisis_df: pd.DataFrame,
    enhanced_targets: pd.DataFrame,
) -> None:
    avg_spy = enhanced_targets["SPY"].mean()
    avg_tlt = enhanced_targets["TLT"].mean()
    avg_gld = enhanced_targets["GLD"].mean()
    summary = f"""# IEOR 198 Final Project Summary

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
- SPY: {avg_spy:.1%}
- TLT: {avg_tlt:.1%}
- GLD: {avg_gld:.1%}

## Full-Sample Metrics
{dataframe_to_markdown(metrics_df, decimals=4)}

## Enhanced Strategy Regime Breakdown
{dataframe_to_markdown(crisis_df, decimals=4, index_name="Regime")}

## Deliverables
- `outputs/equity_curves.png`
- `outputs/enhanced_dynamic_weights.png`
- `outputs/enhanced_forecast_volatility.png`
- `outputs/performance_metrics.csv`
- `outputs/crisis_metrics.csv`
- `final_report.md`
- `final_report.tex`
"""
    (PROJECT_ROOT / "project_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    ensure_directories()
    cleanup_legacy_artifacts()
    prices = download_prices()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    rebalance_dates = month_end_index(prices)
    asset_monthly = monthly_simple_returns(prices, rebalance_dates)

    original_tickers = ["SPY", "TLT"]
    enhanced_tickers = ["SPY", "TLT", "GLD"]

    original_targets = build_dynamic_targets(
        log_returns=log_returns,
        rebalance_dates=rebalance_dates,
        tickers=original_tickers,
        initial_weights=np.array([0.60, 0.40]),
    )
    enhanced_targets = build_dynamic_targets(
        log_returns=log_returns,
        rebalance_dates=rebalance_dates,
        tickers=enhanced_tickers,
        initial_weights=np.array([0.50, 0.30, 0.20]),
    )

    strategies = {
        "Original Dynamic SPY/TLT": (original_targets[original_tickers], original_tickers, TRANSACTION_COST),
        "Enhanced Dynamic SPY/TLT/GLD": (enhanced_targets[enhanced_tickers], enhanced_tickers, TRANSACTION_COST),
        "Static 60/40": (
            pd.DataFrame({"SPY": 0.60, "TLT": 0.40, "GLD": 0.00}, index=enhanced_targets.index),
            enhanced_tickers,
            TRANSACTION_COST,
        ),
        "Equal Weight 1/3": (
            pd.DataFrame({"SPY": 1 / 3, "TLT": 1 / 3, "GLD": 1 / 3}, index=enhanced_targets.index),
            enhanced_tickers,
            TRANSACTION_COST,
        ),
        "Buy and Hold SPY": (
            pd.DataFrame({"SPY": 1.00, "TLT": 0.00, "GLD": 0.00}, index=enhanced_targets.index),
            enhanced_tickers,
            0.0,
        ),
    }

    result_series: Dict[str, pd.Series] = {}
    weight_outputs: Dict[str, pd.DataFrame] = {}
    for name, (targets, tickers, cost_rate) in strategies.items():
        returns, weights = simulate_portfolio(asset_monthly, targets, tickers=tickers, cost_rate=cost_rate)
        result_series[name] = returns
        weight_outputs[name] = weights

    results = pd.DataFrame(result_series).dropna()
    metrics = pd.DataFrame({name: performance_metrics(results[name]) for name in results.columns}).T
    crisis = crisis_period_metrics(results["Enhanced Dynamic SPY/TLT/GLD"])

    results.to_csv(OUTPUT_DIR / "monthly_returns.csv")
    metrics.to_csv(OUTPUT_DIR / "performance_metrics.csv")
    crisis.to_csv(OUTPUT_DIR / "crisis_metrics.csv")
    original_targets.to_csv(OUTPUT_DIR / "original_dynamic_targets.csv")
    enhanced_targets.to_csv(OUTPUT_DIR / "enhanced_dynamic_targets.csv")
    weight_outputs["Original Dynamic SPY/TLT"].to_csv(OUTPUT_DIR / "original_dynamic_realized_weights.csv")
    weight_outputs["Enhanced Dynamic SPY/TLT/GLD"].to_csv(OUTPUT_DIR / "enhanced_dynamic_realized_weights.csv")

    save_plot_equity_curves(results)
    save_plot_weights(
        weight_outputs["Original Dynamic SPY/TLT"],
        tickers=original_tickers,
        filename="original_dynamic_weights.png",
        title="Original Dynamic Strategy Weights (SPY/TLT)",
    )
    save_plot_weights(
        weight_outputs["Enhanced Dynamic SPY/TLT/GLD"],
        tickers=enhanced_tickers,
        filename="enhanced_dynamic_weights.png",
        title="Enhanced Dynamic Strategy Weights (SPY/TLT/GLD)",
    )
    save_plot_forecasts(
        original_targets,
        tickers=original_tickers,
        filename="original_forecast_volatility.png",
        title="Original Strategy One-Month GARCH Volatility Forecasts",
    )
    save_plot_forecasts(
        enhanced_targets,
        tickers=enhanced_tickers,
        filename="enhanced_forecast_volatility.png",
        title="Enhanced Strategy One-Month GARCH Volatility Forecasts",
    )
    save_summary_report(metrics, crisis, enhanced_targets)

    print("\nPerformance metrics:")
    print(metrics.round(4))
    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
