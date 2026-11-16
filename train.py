"""From-scratch MMM: fit, evaluate, decompose, compute ROI and re-allocate the budget.

Run:  python train.py      (about 1 minute; writes everything to outputs/)
"""
# %% Setup
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmm_core import (CHANNELS, SimpleMMM, load_national_data, optimize_budget,
                      save_figure, weekly_response)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# %% 1. Load & explore
df = load_national_data()
print(f"{len(df)} weeks: {df.time.min().date()} -> {df.time.max().date()}")
spend_cols = [f"{c}_spend" for c in CHANNELS]
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(df.time, df.revenue)
ax[0].set_title("Weekly revenue")
ax[1].stackplot(df.time, [df[c] for c in spend_cols], labels=CHANNELS, alpha=0.85)
ax[1].legend(loc="upper left", ncol=5)
ax[1].set_title("Weekly media spend by channel")
save_figure(OUT / "01_eda.png", fig)

# %% 2. Time-based split (70% train / 15% validation / 15% test) and fit
n = len(df)
train_end, val_end = int(n * 0.70), int(n * 0.85)
mmm = SimpleMMM().fit(df, train_end, val_end)


# %% 3. Fit quality
def metrics(actual, predicted):
    return {"R2": 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum(),
            "MAPE": np.mean(np.abs((actual - predicted) / actual))}


y, pred = mmm.y, mmm.predict()
splits = {"train+val": slice(0, val_end), "test": slice(val_end, n)}
fit_quality = {name: {k: round(float(v), 4) for k, v in metrics(y[s], pred[s]).items()}
               for name, s in splits.items()}
for name, m in fit_quality.items():
    print(f"{name:>10}: R2 = {m['R2']:.3f}, MAPE = {m['MAPE']:.1%}")
(OUT / "fit_quality.json").write_text(json.dumps(fit_quality, indent=2))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.time, y, label="actual")
ax.plot(df.time, pred, label="predicted")
ax.axvspan(df.time[val_end], df.time.iloc[-1], alpha=0.15, color="gray", label="test")
ax.legend()
ax.set_title("Actual vs predicted revenue")
save_figure(OUT / "02_fit.png", fig)

# %% 4. Revenue decomposition
dec = mmm.decompose()
fig, ax = plt.subplots(figsize=(12, 5))
ax.stackplot(dec.time, [dec[c] for c in ["baseline", *CHANNELS]],
             labels=["baseline", *CHANNELS], alpha=0.8)
ax.plot(df.time, y, color="black", lw=1, label="actual")
ax.legend(loc="upper left", ncol=4)
ax.set_title("Revenue decomposition")
save_figure(OUT / "03_decomposition.png", fig)
dec.to_csv(OUT / "decomposition.csv", index=False)

# %% 5. ROI by channel (full period, same definition as Meridian)
roi = mmm.roi_table()
print(roi.round(3).to_string(index=False))
roi.to_csv(OUT / "roi_scratch.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(roi.channel, roi.roi)
ax.axhline(1, ls="--", lw=0.8, color="gray")
ax.set_ylabel("ROI (revenue per unit of spend)")
ax.set_title("ROI by channel")
save_figure(OUT / "04_roi.png", fig)

# %% 6. Response curves
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
ax.set_title("Response curves (dot = current spend)")
save_figure(OUT / "05_response_curves.png", fig)

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
budget.to_csv(OUT / "budget_scratch.csv", index=False)
uplift = (opt_rev / cur_rev - 1) * 100
print(budget.round(1).to_string(index=False))
print(f"Same budget, re-allocated: {uplift:+.1f}% media-driven revenue")

# %% 8. Save results and model parameters (used by the Streamlit app)
(OUT / "budget_uplift.json").write_text(json.dumps(
    {"current_weekly_media_revenue": cur_rev, "optimized_weekly_media_revenue": opt_rev,
     "uplift_pct": round(uplift, 2)}, indent=2))
(OUT / "model_params.json").write_text(json.dumps(params, indent=2))
print("Saved charts, CSVs and model_params.json to outputs/")
