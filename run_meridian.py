"""Fit Google Meridian on the same dataset and export results for compare.py.

Needs a GPU for reasonable speed (Google Colab T4: about 15-25 minutes).
Quick smoke test (a few minutes, results not meaningful):
    MERIDIAN_QUICK=1 python run_meridian.py
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from meridian import backend, constants
from meridian.analysis import analyzer, optimizer, summarizer
from meridian.data import data_frame_input_data_builder
from meridian.model import model, prior_distribution, spec

from mmm_core import CHANNELS, load_geo_data

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

QUICK = os.environ.get("MERIDIAN_QUICK") == "1"
ROOT = Path(__file__).resolve().parent
OUT, REP = ROOT / "outputs" / "meridian", ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REP.mkdir(exist_ok=True)

# %% 1. Data: geo-level, impressions + spend (more detail than the national model)
df_geo = load_geo_data()
data = (
    data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type="non_revenue",
        default_kpi_column="conversions",
        default_revenue_per_kpi_column="revenue_per_conversion",
    )
    .with_kpi(df_geo)
    .with_revenue_per_kpi(df_geo)
    .with_population(df_geo)
    .with_controls(df_geo, control_cols=["sentiment_score_control", "competitor_sales_control"])
    .with_media(df_geo,
                media_cols=[f"{c}_impression" for c in CHANNELS],
                media_spend_cols=[f"{c}_spend" for c in CHANNELS],
                media_channels=CHANNELS)
    .with_non_media_treatments(df_geo, non_media_treatment_cols=["Promo"])
    .with_organic_media(df_geo,
                        organic_media_cols=["Organic_channel0_impression"],
                        organic_media_channels=["Organic_channel0"])
    .build()
)

# %% 2. Model: ROI prior LogNormal(0.2, 0.9) (median ROI ~1.2) + automatic knot selection
prior = prior_distribution.PriorDistribution(
    roi_m=backend.tfd.LogNormal(np.float64(0.2), np.float64(0.9), name=constants.ROI_M))
mmm_mer = model.Meridian(input_data=data,
                         model_spec=spec.ModelSpec(prior=prior, enable_aks=True))

# %% 3. Sampling (MCMC)
if QUICK:
    mmm_mer.sample_prior(50, seed=0)
    mmm_mer.sample_posterior(n_chains=1, n_adapt=20, n_burnin=20, n_keep=20, seed=0)
else:
    mmm_mer.sample_prior(500, seed=0)
    mmm_mer.sample_posterior(n_chains=4, n_adapt=500, n_burnin=500, n_keep=500, seed=0)
print("Meridian fitted.")

# %% 4. ROI with 90% credible interval -> outputs/meridian/roi.csv
roi_draws = np.asarray(analyzer.Analyzer(mmm_mer).roi())  # (chains, draws, channels)
flat = roi_draws.reshape(-1, roi_draws.shape[-1])
mer = pd.DataFrame({
    "channel": CHANNELS,
    "roi_meridian": flat.mean(axis=0),
    "roi_low_90": np.quantile(flat, 0.05, axis=0),
    "roi_high_90": np.quantile(flat, 0.95, axis=0),
    "spend": [df_geo[f"{c}_spend"].sum() for c in CHANNELS],
})
mer["incremental_revenue"] = mer.roi_meridian * mer.spend  # ROI x spend, by definition
mer.to_csv(OUT / "roi.csv", index=False)
print(mer.round(3).to_string(index=False))

# %% 5. Budget optimization (same total budget, each channel within +/-50%, like train.py)
opt = optimizer.BudgetOptimizer(mmm_mer).optimize(
    fixed_budget=True, spend_constraint_lower=0.5, spend_constraint_upper=0.5)
cur = np.asarray(opt.nonoptimized_data[constants.SPEND].sel(channel=CHANNELS).values, float)
new = np.asarray(opt.optimized_data[constants.SPEND].sel(channel=CHANNELS).values, float)
budget = pd.DataFrame({"channel": CHANNELS, "current_spend": cur, "optimized_spend": new})
budget["change_pct"] = (budget.optimized_spend / budget.current_spend - 1) * 100
budget.to_csv(OUT / "budget.csv", index=False)
print(budget.round(1).to_string(index=False))

# %% 6. HTML reports -> reports/
times = pd.to_datetime(df_geo["time"])
start, end = times.min().strftime("%Y-%m-%d"), times.max().strftime("%Y-%m-%d")
summarizer.Summarizer(mmm_mer).output_model_results_summary(
    "summary_output.html", str(REP), start, end)
opt.output_optimization_summary("optimization_output.html", str(REP))
try:  # model health card (available in recent Meridian versions)
    from meridian.analysis.review import reviewer
    reviewer.ModelReviewer(mmm_mer).run().output_model_health_card(
        filename="health_card.html", filepath=str(REP))
except Exception as e:  # noqa: BLE001 - the health card is optional
    print("Health card skipped:", type(e).__name__, e)

print("Saved outputs/meridian/roi.csv, outputs/meridian/budget.csv and reports/*.html")
