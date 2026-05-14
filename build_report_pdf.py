from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PDF_PATH = PROJECT_ROOT / "final_report.pdf"


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.95
    fig.text(0.08, y, title, fontsize=18, fontweight="bold", va="top")
    y -= 0.05

    for paragraph in paragraphs:
        wrapped = "\n".join(wrap(paragraph, width=92)) if paragraph else ""
        fig.text(0.08, y, wrapped, fontsize=10.5, va="top")
        line_count = max(wrapped.count("\n") + 1, 1)
        y -= 0.028 * line_count + 0.018

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    image = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    ax.imshow(image)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with PdfPages(PDF_PATH) as pdf:
        add_text_page(
            pdf,
            "Enhanced Volatility-Managed Allocation with GARCH, CVaR, and a Gold Defensive Sleeve",
            [
                "Abstract: This project extends an original SPY/TLT volatility-managed allocation strategy by adding GLD as a third defensive asset. The final model uses monthly walk-forward rebalancing, GARCH(1,1) volatility forecasts, and CVaR-aware optimization to allocate across SPY, TLT, and GLD.",
                "Key idea: The original SPY/TLT proposal was vulnerable when stocks and long-duration Treasuries sold off together. Adding gold improves the robustness of the defensive sleeve and leads to better risk-adjusted performance.",
                "Headline result: The enhanced SPY/TLT/GLD strategy improves Sharpe ratio from 0.889 to 0.991, cuts max drawdown from -28.27% to -22.67%, and lowers monthly CVaR 95% from 5.70% to 4.63% relative to the original dynamic model.",
            ],
        )

        add_text_page(
            pdf,
            "Methodology and Results",
            [
                "Dataset: Daily adjusted close prices for SPY, TLT, and GLD from 2008-01-01 to 2024-12-31.",
                "Walk-forward design: Monthly rebalancing with a trailing 756-day estimation window.",
                "Risk model: Per-asset GARCH(1,1) one-month volatility forecast with an EWMA fallback.",
                "Allocation rule: Choose weights that maximize expected return minus a CVaR penalty and turnover penalty, with SPY constrained between 10% and 90%.",
                "Performance summary: Enhanced Dynamic SPY/TLT/GLD achieved cumulative return 2.3750, annualized return 9.08%, annualized volatility 9.16%, Sharpe ratio 0.9913, max drawdown -22.67%, and monthly CVaR 95% 4.63%.",
                "Interpretation: The enhanced model does not maximize absolute return versus buy-and-hold SPY, but it delivers the strongest overall risk-adjusted profile among the tested strategies and handles the 2022-2024 regime substantially better than the original design.",
            ],
        )

        add_text_page(
            pdf,
            "DRL Extension",
            [
                "The repository also includes a PPO-based deep reinforcement learning extension for monthly asset allocation across SPY, TLT, and GLD.",
                "State design: trailing multi-horizon returns, realized volatilities, one-month GARCH forecasts, rolling correlations, and previous portfolio weights.",
                "Action design: a continuous three-asset weight vector normalized to sum to one, with the same long-only and SPY allocation constraints used in the rule-based baseline.",
                "Reward design: realized portfolio return minus volatility and turnover penalties.",
                "Result summary: In the strict 2021-2024 out-of-sample test window, the DRL policy produced stronger return and Sharpe ratio than the rule-based baselines, but its tail risk remained worse than the enhanced GARCH-CVaR allocator. This makes the DRL result promising, but not yet a full replacement for the more interpretable baseline.",
            ],
        )

        add_image_page(pdf, "Equity Curves", OUTPUT_DIR / "equity_curves.png")
        add_image_page(pdf, "Enhanced Dynamic Weights", OUTPUT_DIR / "enhanced_dynamic_weights.png")
        add_image_page(pdf, "Enhanced Forecast Volatility", OUTPUT_DIR / "enhanced_forecast_volatility.png")
        add_image_page(pdf, "DRL Equity Curves", OUTPUT_DIR / "drl_equity_curves.png")


if __name__ == "__main__":
    main()
