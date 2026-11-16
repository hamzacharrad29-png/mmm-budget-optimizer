"""From-scratch MMM at national and regional level: fit, evaluate, decompose,
compute ROI and re-allocate the budget.

Run:  python train.py                      (both levels, about 3 minutes)
      python train.py --level regional     (one level only)
Results go to outputs/national/ and outputs/regional/.
"""
# %% Setup
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmm_core import (CHANNELS, LEVELS, MMM, load_panel, optimize_budget,
                      save_figure, weekly_response)

OUT = Path(__file__).resolve().parent / "outputs"
TEST_SHARE = 0.15  # last 15% of weeks = hold-out test set

parser = argparse.ArgumentParser()
parser.add_argument("--level", choices=[*LEVELS, "both"], default="both")
args, _ = parser.parse_known_args()  # parse_known_args: also works with %run in notebooks
levels = LEVELS if args.level == "both" else [args.level]

# %% 1. Explore (national totals)
nat = load_panel("national")
print(f"{nat.shape[1]} weeks: {nat.times.min().date()} -> {nat.times.max().date()}, "
      f"{load_panel('regional').shape[0]} regions")
OUT.mkdir(exist_ok=True)
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(nat.times, nat.revenue[0])
ax[0].set_title("Weekly revenue (all regions)")
ax[1].stackplot(nat.times, nat.spend[:, 0, :], labels=CHANNELS, alpha=0.85)
ax[1].legend(loc="upper left", ncol=5)
ax[1].set_title("Weekly media spend by channel")
save_figure(OUT / "01_eda.png", fig)


def run(level: str) -> None:
    out = OUT / level
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {level.upper()} MODEL =====")

    # %% 2. Fit on the first 85% of weeks, keep the last 15% as test set
    panel = load_panel(level)
    T = panel.shape[1]
    fit_end = int(T * (1 - TEST_SHARE))
    mmm = MMM().fit(panel, fit_end)

    # %% 3. Fit quality
    quality = mmm.fit_quality()
    for scope, q in quality.items():
        for window, m in q.items():
            print(f"{scope:>15} {window:>4}: R2 = {m['R2']:.3f}, MAPE = {m['MAPE']:.1%}")
    (out / "fit_quality.json").write_text(json.dumps(quality, indent=2))

    actual, pred = panel.revenue.sum(0), mmm.predict().sum(0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(panel.times, actual, label="actual")
    ax.plot(panel.times, pred, label="predicted")
    ax.axvspan(panel.times[fit_end], panel.times[-1], alpha=0.15, color="gray", label="test")
    ax.legend()
    ax.set_title(f"Actual vs predicted revenue, all regions ({level} model)")
    save_figure(out / "02_fit.png", fig)

    # %% 4. Revenue decomposition
    dec = mmm.decompose()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(dec.time, [dec[c] for c in ["baseline", *CHANNELS]],
                 labels=["baseline", *CHANNELS], alpha=0.8)
    ax.plot(panel.times, actual, color="black", lw=1, label="actual")
    ax.legend(loc="upper left", ncol=4)
    ax.set_title(f"Revenue decomposition ({level} model)")
    save_figure(out / "03_decomposition.png", fig)
    dec.to_csv(out / "decomposition.csv", index=False)

    # %% 5. ROI by channel (full period, same definition as Meridian)
    roi = mmm.roi_table()
    print(roi.round(3).to_string(index=False))
    roi.to_csv(out / "roi.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(roi.channel, roi.roi)
    ax.axhline(1, ls="--", lw=0.8, color="gray")
    ax.set_ylabel("ROI (revenue per unit of spend)")
    ax.set_title(f"ROI by channel ({level} model)")
    save_figure(out / "04_roi.png", fig)

    # %% 6. Response curves (total weekly spend, split across regions as historically)
    params = mmm.channel_params()
    fig, ax = plt.subplots(figsize=(10, 5))
    for ch, p in params.items():
        grid = np.linspace(0, 2 * p["avg_weekly_spend"], 100)
        line, = ax.plot(grid, weekly_response(grid, p), label=ch)
        ax.scatter(p["avg_weekly_spend"], weekly_response(p["avg_weekly_spend"], p),
                   color=line.get_color())
    ax.set_xlabel("Weekly spend")
    ax.set_ylabel("Incremental weekly revenue")
    ax.legend()
    ax.set_title(f"Response curves, dot = current spend ({level} model)")
    save_figure(out / "05_response_curves.png", fig)

    # %% 7. Budget re-allocation at constant total budget (each channel within +/-50%)
    total = sum(p["avg_weekly_spend"] for p in params.values())
    alloc, opt_rev = optimize_budget(params, total)
    cur_rev = sum(float(weekly_response(p["avg_weekly_spend"], p)) for p in params.values())
    budget = pd.DataFrame({
        "channel": list(params),
        "current_weekly_spend": [params[c]["avg_weekly_spend"] for c in params],
        "optimized_weekly_spend": [alloc[c] for c in params],
    })
    budget["change_pct"] = (budget.optimized_weekly_spend / budget.current_weekly_spend - 1) * 100
    budget.to_csv(out / "budget.csv", index=False)
    uplift = (opt_rev / cur_rev - 1) * 100
    print(budget.round(1).to_string(index=False))
    print(f"Same budget, re-allocated: {uplift:+.1f}% media-driven revenue")

    # %% 8. Save results and model parameters (used by the Streamlit app)
    (out / "budget_uplift.json").write_text(json.dumps(
        {"current_weekly_media_revenue": cur_rev, "optimized_weekly_media_revenue": opt_rev,
         "uplift_pct": round(uplift, 2)}, indent=2))
    (out / "model_params.json").write_text(json.dumps(params, indent=2))
    print(f"Saved charts, CSVs and model_params.json to outputs/{level}/")


for level in levels:
    run(level)
