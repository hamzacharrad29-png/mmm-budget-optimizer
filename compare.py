"""Compare the from-scratch MMM (national and regional) with Google Meridian.

Needs outputs/national/, outputs/regional/ (from train.py) and outputs/meridian/ (from run_meridian.py).
Writes outputs/comparison/.

Run:  python compare.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmm_core import LEVELS, save_figure

OUT = Path(__file__).resolve().parent / "outputs"
CMP = OUT / "comparison"
needed = [OUT / lvl / "roi.csv" for lvl in LEVELS] + [OUT / "meridian" / "roi.csv"]
missing = [str(p.relative_to(OUT.parent)) for p in needed if not p.exists()]
if missing:
    sys.exit(f"Missing {', '.join(missing)}. Run train.py and run_meridian.py first.")
CMP.mkdir(exist_ok=True)

mer = pd.read_csv(OUT / "meridian" / "roi.csv")
comp = mer[["channel", "roi_meridian", "roi_low_90", "roi_high_90"]].copy()
for lvl in LEVELS:
    comp[f"roi_{lvl}"] = pd.read_csv(OUT / lvl / "roi.csv")["roi"].to_numpy()
comp.to_csv(CMP / "roi_comparison.csv", index=False)

mer_budget = pd.read_csv(OUT / "meridian" / "budget.csv")


def direction(pct):
    return np.where(pct > 1, 1, np.where(pct < -1, -1, 0))


rows = []
for lvl in LEVELS:
    roi = comp[f"roi_{lvl}"]
    budget = pd.read_csv(OUT / lvl / "budget.csv")
    rows.append({
        "model": f"from-scratch {lvl}",
        # Spearman rank correlation with Meridian's posterior mean ROI
        "rank_correlation": round(roi.rank().corr(comp.roi_meridian.rank()), 2),
        "roi_inside_meridian_90pct": int(roi.between(comp.roi_low_90, comp.roi_high_90).sum()),
        "same_budget_direction": int((direction(budget.change_pct) ==
                                      direction(mer_budget.change_pct)).sum()),
        "best_channel": comp.channel[roi.idxmax()],
    })
summary = pd.DataFrame(rows)
summary.to_csv(CMP / "summary.csv", index=False)

budget_cmp = mer_budget[["channel"]].assign(
    **{f"change_{lvl}_pct": pd.read_csv(OUT / lvl / "budget.csv").change_pct for lvl in LEVELS},
    change_meridian_pct=mer_budget.change_pct)
budget_cmp.to_csv(CMP / "budget_comparison.csv", index=False)

print("ROI by channel:")
print(comp.round(2).to_string(index=False))
print(f"\nAgreement with Meridian (out of {len(comp)} channels; Meridian's best: "
      f"{comp.channel[comp.roi_meridian.idxmax()]}):")
print(summary.to_string(index=False))
print("\nBudget recommendation (% change in spend, same total budget, +/-50% per channel):")
print(budget_cmp.round(1).to_string(index=False))

# Chart: ROI by channel for the three models, with Meridian's 90% credible interval
x = np.arange(len(comp))
w = 0.27
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w, comp.roi_national, w, label="From-scratch, national", color="#9db4d6")
ax.bar(x, comp.roi_regional, w, label="From-scratch, regional", color="#2f6db5")
ax.bar(x + w, comp.roi_meridian, w, label="Meridian (posterior mean)", color="#f28e2b",
       yerr=[(comp.roi_meridian - comp.roi_low_90).clip(lower=0),
             (comp.roi_high_90 - comp.roi_meridian).clip(lower=0)], capsize=4)
ax.set_xticks(x, comp.channel)
ax.set_ylabel("ROI")
ax.axhline(1, ls="--", lw=0.8, color="gray")
ax.set_title("ROI by channel (Meridian error bars = 90% credible interval)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
save_figure(CMP / "roi_comparison.png", fig)
