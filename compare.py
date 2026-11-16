"""Compare the from-scratch MMM with Google Meridian.

Needs outputs/roi_scratch.csv (from train.py) and outputs/roi_meridian.csv (from run_meridian.py).
If outputs/budget_scratch.csv and outputs/budget_meridian.csv exist, budget directions are compared too.

Run:  python compare.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmm_core import save_figure

OUT = Path(__file__).resolve().parent / "outputs"
for f in ("roi_scratch.csv", "roi_meridian.csv"):
    if not (OUT / f).exists():
        sys.exit(f"Missing outputs/{f}. Run train.py and run_meridian.py first.")

mine = pd.read_csv(OUT / "roi_scratch.csv")
mer = pd.read_csv(OUT / "roi_meridian.csv")

comp = (
    mine[["channel", "roi", "incremental_revenue"]]
    .rename(columns={"roi": "roi_scratch", "incremental_revenue": "inc_scratch"})
    .merge(mer[["channel", "roi_meridian", "roi_low_90", "roi_high_90", "incremental_revenue"]]
           .rename(columns={"incremental_revenue": "inc_meridian"}), on="channel")
)

# 1. Does the from-scratch ROI fall inside Meridian's 90% credible interval?
comp["inside_interval"] = comp.roi_scratch.between(comp.roi_low_90, comp.roi_high_90)

# 2. Channel ranking (1 = best ROI); correlation of ranks = Spearman correlation
comp["rank_scratch"] = comp.roi_scratch.rank(ascending=False, method="min").astype(int)
comp["rank_meridian"] = comp.roi_meridian.rank(ascending=False, method="min").astype(int)
rank_corr = comp.rank_scratch.corr(comp.rank_meridian)

# 3. Share of media-driven revenue
comp["share_scratch"] = comp.inc_scratch / comp.inc_scratch.sum()
comp["share_meridian"] = comp.inc_meridian / comp.inc_meridian.sum()

print(comp.round(3).to_string(index=False))
print(f"\nRank correlation: {rank_corr:.2f}  (1 = same ranking, 0 = unrelated)")
print(f"Channels inside Meridian's 90% interval: {comp.inside_interval.sum()}/{len(comp)}")
comp.to_csv(OUT / "model_comparison.csv", index=False)

# 4. Budget recommendation: same direction (increase / decrease)?
if (OUT / "budget_scratch.csv").exists() and (OUT / "budget_meridian.csv").exists():
    b = (pd.read_csv(OUT / "budget_scratch.csv")[["channel", "change_pct"]]
         .rename(columns={"change_pct": "change_scratch_pct"})
         .merge(pd.read_csv(OUT / "budget_meridian.csv")[["channel", "change_pct"]]
                .rename(columns={"change_pct": "change_meridian_pct"}), on="channel"))

    def direction(v):
        return "up" if v > 1 else ("down" if v < -1 else "flat")

    b["dir_scratch"] = b.change_scratch_pct.map(direction)
    b["dir_meridian"] = b.change_meridian_pct.map(direction)
    b["agree"] = b.dir_scratch == b.dir_meridian
    print("\nBudget recommendation (% change in spend, same total budget):")
    print(b.round(1).to_string(index=False))
    print(f"Same direction on {b.agree.sum()}/{len(b)} channels")
    b.to_csv(OUT / "budget_comparison.csv", index=False)

# Chart: ROI side by side, with Meridian's 90% credible interval
x = np.arange(len(comp))
w = 0.38
err_low = (comp.roi_meridian - comp.roi_low_90).clip(lower=0)
err_high = (comp.roi_high_90 - comp.roi_meridian).clip(lower=0)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w / 2, comp.roi_scratch, w, label="From-scratch model")
ax.bar(x + w / 2, comp.roi_meridian, w, label="Meridian (posterior mean)",
       yerr=[err_low, err_high], capsize=4)
ax.set_xticks(x, comp.channel)
ax.set_ylabel("ROI")
ax.axhline(1, ls="--", lw=0.8, color="gray")
ax.set_title("ROI by channel: from-scratch vs Meridian (error bars = 90% credible interval)")
ax.legend()
save_figure(OUT / "06_model_comparison.png", fig)
