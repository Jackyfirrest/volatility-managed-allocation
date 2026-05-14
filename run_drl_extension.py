from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_final_project import ASSETS, DATA_DIR, OUTPUT_DIR, garch_forecast_vol, performance_metrics


TRAIN_START = "2011-01-31"
TRAIN_END = "2017-12-29"
VALID_START = "2018-01-31"
VALID_END = "2020-12-31"
TEST_START = "2021-01-29"
TEST_END = "2024-12-31"

TRANSACTION_COST = 0.001
RISK_PENALTY = 0.15
TURNOVER_PENALTY = 0.002
SPY_BOUNDS = (0.10, 0.90)


@dataclass
class RolloutResult:
    returns: pd.Series
    weights: pd.DataFrame


def month_end_index(prices: pd.DataFrame) -> pd.DatetimeIndex:
    grouped = prices.groupby([prices.index.year, prices.index.month])
    return pd.DatetimeIndex(grouped.tail(1).index)


def monthly_simple_returns(prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    records = []
    for idx in range(len(rebalance_dates) - 1):
        start = rebalance_dates[idx]
        end = rebalance_dates[idx + 1]
        period = prices.loc[[start, end]]
        simple_ret = period.iloc[-1] / period.iloc[0] - 1.0
        records.append({"Date": end, **simple_ret.to_dict()})
    return pd.DataFrame.from_records(records).set_index("Date")


def normalize_weights(raw_action: np.ndarray) -> np.ndarray:
    action = np.clip(raw_action.astype(float), 0.0, 1.0)
    if action.sum() <= 0:
        weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        weights = action / action.sum()

    spy_weight = weights[0]
    if spy_weight < SPY_BOUNDS[0]:
        remainder = weights[1:] / max(weights[1:].sum(), 1e-8)
        weights[0] = SPY_BOUNDS[0]
        weights[1:] = (1.0 - SPY_BOUNDS[0]) * remainder
    elif spy_weight > SPY_BOUNDS[1]:
        remainder = weights[1:] / max(weights[1:].sum(), 1e-8)
        weights[0] = SPY_BOUNDS[1]
        weights[1:] = (1.0 - SPY_BOUNDS[1]) * remainder
    return weights


def build_monthly_features(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_log_returns = np.log(prices / prices.shift(1)).dropna()
    rebalance_dates = month_end_index(prices)
    monthly_returns = monthly_simple_returns(prices, rebalance_dates)
    feature_rows: List[Dict[str, float]] = []

    for date in monthly_returns.index:
        if date not in prices.index:
            continue
        daily_window = daily_log_returns.loc[:date].tail(252)
        if len(daily_window) < 126:
            continue
        monthly_window = monthly_returns.loc[:date]
        if len(monthly_window) < 12:
            continue

        features: Dict[str, float] = {"Date": date}
        for ticker in ASSETS:
            series = monthly_window[ticker]
            features[f"{ticker}_ret_1m"] = float(series.tail(1).mean())
            features[f"{ticker}_ret_3m"] = float(series.tail(3).mean())
            features[f"{ticker}_ret_6m"] = float(series.tail(6).mean())
            features[f"{ticker}_ret_12m"] = float(series.tail(12).mean())
            features[f"{ticker}_vol_3m"] = float(series.tail(3).std(ddof=1))
            features[f"{ticker}_vol_6m"] = float(series.tail(6).std(ddof=1))
            features[f"{ticker}_vol_12m"] = float(series.tail(12).std(ddof=1))
            features[f"{ticker}_garch_1m"] = float(garch_forecast_vol(daily_window[ticker]))

        corr = monthly_window[ASSETS].tail(12).corr()
        features["corr_spy_tlt"] = float(corr.loc["SPY", "TLT"])
        features["corr_spy_gld"] = float(corr.loc["SPY", "GLD"])
        features["corr_tlt_gld"] = float(corr.loc["TLT", "GLD"])
        feature_rows.append(features)

    feature_df = pd.DataFrame.from_records(feature_rows).set_index("Date").sort_index()
    monthly_returns = monthly_returns.loc[feature_df.index].copy()
    return feature_df, monthly_returns


class AllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_df: pd.DataFrame,
        monthly_returns: pd.DataFrame,
        risk_penalty: float = RISK_PENALTY,
        turnover_penalty: float = TURNOVER_PENALTY,
    ) -> None:
        super().__init__()
        self.features = feature_df.copy()
        self.returns = monthly_returns.loc[self.features.index].copy()
        self.risk_penalty = risk_penalty
        self.turnover_penalty = turnover_penalty
        self.feature_cols = list(self.features.columns)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.feature_cols) + len(ASSETS),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(len(ASSETS),), dtype=np.float32)
        self.current_step = 0
        self.previous_weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        self.feature_mean = self.features.mean()
        self.feature_std = self.features.std(ddof=1).replace(0.0, 1.0)

    def _observation(self) -> np.ndarray:
        raw = self.features.iloc[self.current_step]
        scaled = ((raw - self.feature_mean) / self.feature_std).fillna(0.0).to_numpy(dtype=np.float32)
        prev = self.previous_weights.astype(np.float32)
        return np.concatenate([scaled, prev]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        return self._observation(), {}

    def step(self, action: np.ndarray):
        weights = normalize_weights(action)
        realized = self.returns.iloc[self.current_step].to_numpy(dtype=float)
        turnover = float(np.abs(weights - self.previous_weights).sum())
        portfolio_return = float(np.dot(weights, realized))
        portfolio_vol = float(np.std(realized))
        reward = portfolio_return - self.risk_penalty * portfolio_vol - self.turnover_penalty * turnover
        self.previous_weights = weights

        self.current_step += 1
        terminated = self.current_step >= len(self.features)
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._observation()

        info = {
            "weights": weights,
            "portfolio_return": portfolio_return,
            "turnover": turnover,
        }
        return obs, reward, terminated, False, info


def slice_period(
    feature_df: pd.DataFrame, monthly_returns: pd.DataFrame, start: str, end: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = feature_df.loc[start:end].copy()
    returns = monthly_returns.loc[features.index].copy()
    return features, returns


def run_policy(
    model: PPO, feature_df: pd.DataFrame, monthly_returns: pd.DataFrame
) -> RolloutResult:
    env = AllocationEnv(feature_df, monthly_returns)
    obs, _ = env.reset()
    returns: List[float] = []
    weights_rows: List[Dict[str, float]] = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, _, terminated, _, info = env.step(action)
        date = feature_df.index[env.current_step - 1]
        returns.append(info["portfolio_return"] - TRANSACTION_COST * info["turnover"])
        weights_rows.append({"Date": date, **{asset: info["weights"][idx] for idx, asset in enumerate(ASSETS)}})
        obs = next_obs
        if terminated:
            break

    return RolloutResult(
        returns=pd.Series(returns, index=feature_df.index, name="DRL PPO"),
        weights=pd.DataFrame.from_records(weights_rows).set_index("Date"),
    )


def evaluate_constant_weight(monthly_returns: pd.DataFrame, weights: np.ndarray, name: str) -> pd.Series:
    previous = weights.copy()
    out: List[float] = []
    for _, row in monthly_returns.iterrows():
        turnover = float(np.abs(weights - previous).sum())
        gross = float(np.dot(weights, row.to_numpy(dtype=float)))
        out.append(gross - TRANSACTION_COST * turnover)
        end_vals = weights * (1.0 + row.to_numpy(dtype=float))
        previous = end_vals / end_vals.sum()
    return pd.Series(out, index=monthly_returns.index, name=name)


def save_drl_plot(results: pd.DataFrame) -> None:
    wealth = (1.0 + results).cumprod()
    plt.figure(figsize=(11, 6))
    for column in wealth.columns:
        plt.plot(wealth.index, wealth[column], label=column, linewidth=2)
    plt.title("DRL Extension: Out-of-Sample Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drl_equity_curves.png", dpi=200)
    plt.close()


def save_drl_weights_plot(weights: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 5))
    for asset in ASSETS:
        plt.plot(weights.index, weights[asset], label=f"{asset} weight", linewidth=2)
    plt.title("DRL PPO Test Weights Over Time")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drl_test_weights.png", dpi=200)
    plt.close()


def load_baseline_test_series(start: str, end: str) -> Dict[str, pd.Series]:
    monthly_results = pd.read_csv(OUTPUT_DIR / "monthly_returns.csv", index_col=0, parse_dates=True)
    test_window = monthly_results.loc[start:end].copy()
    return {
        "Original Dynamic SPY/TLT (Test)": test_window["Original Dynamic SPY/TLT"],
        "Enhanced Dynamic SPY/TLT/GLD (Test)": test_window["Enhanced Dynamic SPY/TLT/GLD"],
        "Static 60/40 (Test)": test_window["Static 60/40"],
        "Equal Weight 1/3 (Test)": test_window["Equal Weight 1/3"],
        "Buy and Hold SPY (Test)": test_window["Buy and Hold SPY"],
    }


def main() -> None:
    prices = pd.read_csv(DATA_DIR / "adjusted_close.csv", index_col=0, parse_dates=True)
    feature_df, monthly_returns = build_monthly_features(prices[ASSETS])

    train_features, train_returns = slice_period(feature_df, monthly_returns, TRAIN_START, TRAIN_END)
    valid_features, valid_returns = slice_period(feature_df, monthly_returns, VALID_START, VALID_END)
    test_features, test_returns = slice_period(feature_df, monthly_returns, TEST_START, TEST_END)

    train_env = DummyVecEnv([lambda: AllocationEnv(train_features, train_returns)])
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        n_steps=24,
        batch_size=24,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        seed=42,
    )
    model.learn(total_timesteps=20_000)

    valid_rollout = run_policy(model, valid_features, valid_returns)
    test_rollout = run_policy(model, test_features, test_returns)

    baseline_test_results = load_baseline_test_series(TEST_START, TEST_END)
    diagnostic_results = {
        "DRL PPO (Validation)": valid_rollout.returns,
        "DRL PPO (Test)": test_rollout.returns,
        "Equal Weight 1/3 (Test, Recomputed)": evaluate_constant_weight(test_returns, np.array([1 / 3, 1 / 3, 1 / 3]), "Equal Weight"),
        "Static 60/40/0 (Test, Recomputed)": evaluate_constant_weight(test_returns, np.array([0.60, 0.40, 0.00]), "Static 60/40/0"),
        "Buy and Hold SPY (Test, Recomputed)": evaluate_constant_weight(test_returns, np.array([1.00, 0.00, 0.00]), "SPY"),
    }
    comparison_results = {"DRL PPO (Test)": test_rollout.returns, **baseline_test_results}

    results = pd.DataFrame(comparison_results).dropna(how="all")
    metrics = pd.DataFrame({name: performance_metrics(series.dropna()) for name, series in diagnostic_results.items()}).T
    comparison_metrics = pd.DataFrame({name: performance_metrics(series.dropna()) for name, series in comparison_results.items()}).T

    valid_rollout.returns.to_csv(OUTPUT_DIR / "drl_validation_returns.csv")
    test_rollout.returns.to_csv(OUTPUT_DIR / "drl_test_returns.csv")
    valid_rollout.weights.to_csv(OUTPUT_DIR / "drl_validation_weights.csv")
    test_rollout.weights.to_csv(OUTPUT_DIR / "drl_test_weights.csv")
    metrics.to_csv(OUTPUT_DIR / "drl_performance_metrics.csv")
    comparison_metrics.to_csv(OUTPUT_DIR / "drl_vs_baseline_metrics.csv")
    save_drl_plot(results.dropna())
    save_drl_weights_plot(test_rollout.weights)

    print("\nDRL extension metrics:")
    print(metrics.round(4))
    print("\nDRL vs baseline test comparison:")
    print(comparison_metrics.round(4))
    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
