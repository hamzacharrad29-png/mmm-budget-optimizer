"""Streamlit budget allocation simulator.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mmm_core import optimize_budget, weekly_response

st.set_page_config(page_title="MMM Budget Optimizer", layout="wide")

OUTPUTS = Path(__file__).resolve().parent / "outputs"

with st.sidebar:
    st.header("Scenario")
    level = st.radio("Model", ["Regional (40 regions)", "National"], index=0,
                     help="Same model, fitted on regional data or on national totals.")
    level = "regional" if level.startswith("Regional") else "national"
    budget_pct = st.slider("Total weekly budget (% of current)", 60, 140, 100, 5)
    band = st.slider("Max change per channel (+/- %)", 10, 80, 50, 5)

PARAMS = OUTPUTS / level / "model_params.json"
if not PARAMS.exists():
    st.error(f"Model parameters not found. Run `python train.py` first to create outputs/{level}/model_params.json.")
    st.stop()
params = json.loads(PARAMS.read_text())
channels = list(params)
current = np.array([params[c]["avg_weekly_spend"] for c in channels])


def response(spends):
    return np.array([float(weekly_response(s, params[c])) for s, c in zip(spends, channels)])


st.title("Marketing Mix Modeling: Budget Allocation Simulator")
st.caption(f"Geometric adstock + Hill saturation model ({level} level) fitted on Google Meridian's "
           "simulated dataset. Revenue figures are steady-state weekly media-driven revenue.")

lo, hi = 1 - band / 100, 1 + band / 100
if not lo <= budget_pct / 100 <= hi:
    st.warning("The budget change is larger than the allowed per-channel change. "
               "Widen the per-channel range in the sidebar.")
    st.stop()

total = current.sum() * budget_pct / 100
alloc, _ = optimize_budget(params, total, lo, hi)
optimized = np.array([alloc[c] for c in channels])
same_mix = current * budget_pct / 100
rev_same, rev_opt = response(same_mix), response(optimized)

c1, c2, c3 = st.columns(3)
c1.metric("Weekly budget", f"{total:,.0f}")
c2.metric("Media revenue, current mix", f"{rev_same.sum():,.0f}")
c3.metric("Media revenue, optimized mix", f"{rev_opt.sum():,.0f}",
          f"{(rev_opt.sum() / rev_same.sum() - 1) * 100:+.1f}%")

fig = go.Figure([go.Bar(name="Current mix", x=channels, y=same_mix),
                 go.Bar(name="Optimized", x=channels, y=optimized)])
fig.update_layout(barmode="group", title="Weekly spend by channel", height=380)
st.plotly_chart(fig)

eps = 0.01
mroi = (response(optimized * (1 + eps)) - rev_opt) / (optimized * eps)
table = pd.DataFrame({
    "Channel": channels,
    "Current spend": same_mix,
    "Optimized spend": optimized,
    "Change %": (optimized / same_mix - 1) * 100,
    "ROI": rev_opt / optimized,
    "Marginal ROI": mroi,
})
st.dataframe(table.style.format({"Current spend": "{:,.0f}", "Optimized spend": "{:,.0f}",
                                 "Change %": "{:+.1f}", "ROI": "{:.2f}", "Marginal ROI": "{:.2f}"}),
             hide_index=True)

st.subheader("Response curve")
ch = st.selectbox("Channel", channels)
i = channels.index(ch)
grid = np.linspace(0, 2 * current[i], 150)
curve = go.Figure(go.Scatter(x=grid, y=weekly_response(grid, params[ch]), name="response"))
curve.add_scatter(x=[same_mix[i]], y=[rev_same[i]], mode="markers", name="current", marker_size=12)
curve.add_scatter(x=[optimized[i]], y=[rev_opt[i]], mode="markers", name="optimized", marker_size=12)
curve.update_layout(xaxis_title="Weekly spend", yaxis_title="Incremental weekly revenue", height=380)
st.plotly_chart(curve)
st.caption("At the optimum, marginal ROI is roughly equal across channels that are not at a bound.")
