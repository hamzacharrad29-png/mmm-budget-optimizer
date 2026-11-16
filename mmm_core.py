"""Lightweight Marketing Mix Model.

Geometric adstock + Hill saturation + bounded least squares, with the media
hyperparameters tuned by differential evolution on a validation window.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, lsq_linear, minimize

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "geo_all_channels.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/google/meridian/refs/heads/main/"
    "meridian/data/simulated_data/csv/geo_all_channels.csv"
)
CHANNELS = [f"Channel{i}" for i in range(5)]
CONTROLS = ["competitor_sales_control", "sentiment_score_control", "Promo"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_geo_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Meridian's simulated geo-level dataset (one row per geo x week).

    Uses the copy in data/ so results are reproducible; downloads it if missing.
    """
    if not Path(path).exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL, index_col=0).to_csv(path)
    return pd.read_csv(path, index_col=0)


def load_national_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Aggregate the geo data to national weekly level, with revenue = conversions x revenue/conversion."""
    df = load_geo_data(path)
    df["revenue"] = df["conversions"] * df["revenue_per_conversion"]
    spend_cols = [f"{c}_spend" for c in CHANNELS]
    agg = {"revenue": "sum", "conversions": "sum",
           **{c: "sum" for c in spend_cols},
           **{c: "mean" for c in CONTROLS}}
    nat = df.groupby("time", as_index=False).agg(agg)
    nat["time"] = pd.to_datetime(nat["time"])
    return nat.sort_values("time").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Media transformations
# ----------------------------------------------------------------------------
def geometric_adstock(x, decay: float) -> np.ndarray:
    """Carry-over effect: a_t = x_t + decay * a_{t-1}."""
    out = np.empty(len(x))
    carry = 0.0
    for t, v in enumerate(x):
        carry = v + decay * carry
        out[t] = carry
    return out


def hill(x, k: float, a: float):
    """Diminishing returns. k = half-saturation point, a = shape."""
    x = np.clip(x, 0, None)
    return x**a / (x**a + k**a)


def media_feature(spend, scale, decay, k, a) -> np.ndarray:
    return hill(geometric_adstock(np.asarray(spend, float) / scale, decay), k, a)


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Intercept, trend, yearly seasonality (Fourier terms) and standardized controls."""
    n = len(df)
    t = np.arange(n)
    cols = {"intercept": np.ones(n), "trend": t / n}
    for order in (1, 2):
        cols[f"sin{order}"] = np.sin(2 * np.pi * order * t / 52.18)
        cols[f"cos{order}"] = np.cos(2 * np.pi * order * t / 52.18)
    for c in CONTROLS:
        v = df[c].to_numpy(float)
        sd = v.std()
        cols[c] = (v - v.mean()) / sd if sd > 0 else np.zeros(n)
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class SimpleMMM:
    """Adstock + saturation regression with non-negative media coefficients."""

    # Search ranges per channel: adstock decay, Hill half-saturation k, Hill shape a
    BOUNDS = [(0.0, 0.9), (0.1, 3.0), (0.5, 3.0)]

    def __init__(self, channels=CHANNELS):
        self.channels = list(channels)

    def _media(self, theta) -> np.ndarray:
        cols = []
        for i, ch in enumerate(self.channels):
            d, k, a = theta[3 * i: 3 * i + 3]
            cols.append(media_feature(self.df[f"{ch}_spend"], self.scales[ch], d, k, a))
        return np.column_stack(cols)

    def _solve(self, X, y) -> np.ndarray:
        """Least squares with media coefficients forced to be >= 0."""
        lb = np.r_[np.full(self.n_base, -np.inf), np.zeros(X.shape[1] - self.n_base)]
        return lsq_linear(X, y, bounds=(lb, np.full(X.shape[1], np.inf))).x

    def fit(self, df, train_end: int, val_end: int, maxiter: int = 40, seed: int = 42):
        """Tune hyperparameters on [train_end, val_end), then refit coefficients on [0, val_end).

        Rows from val_end onwards are never used for fitting (hold-out test set).
        """
        self.df = df.reset_index(drop=True)
        self.y = self.df["revenue"].to_numpy(float)
        # Spend is scaled by its maximum over the fitting window only (no test leakage)
        self.scales = {ch: float(self.df[f"{ch}_spend"].iloc[:val_end].max()) or 1.0
                       for ch in self.channels}
        base = base_features(self.df)
        self.base_cols, self.base, self.n_base = list(base.columns), base.values, base.shape[1]
        tr, va = slice(0, train_end), slice(train_end, val_end)
        y = self.y

        def objective(theta):
            X = np.hstack([self.base, self._media(theta)])
            beta = self._solve(X[tr], y[tr])
            err = y[va] - X[va] @ beta
            return np.sqrt(np.mean(err**2)) / (y[va].max() - y[va].min())  # NRMSE

        res = differential_evolution(objective, self.BOUNDS * len(self.channels),
                                     maxiter=maxiter, seed=seed, polish=False)
        self.theta = res.x
        self.X = np.hstack([self.base, self._media(self.theta)])
        self.beta = self._solve(self.X[:val_end], y[:val_end])
        return self

    def predict(self) -> np.ndarray:
        return self.X @ self.beta

    def decompose(self) -> pd.DataFrame:
        """Weekly revenue split into baseline and per-channel contributions."""
        nb = self.n_base
        dec = pd.DataFrame(self.X[:, nb:] * self.beta[nb:], columns=self.channels)
        dec.insert(0, "baseline", self.X[:, :nb] @ self.beta[:nb])
        dec["time"] = self.df["time"].values
        return dec

    def roi_table(self, rows=slice(None)) -> pd.DataFrame:
        dec, d = self.decompose().iloc[rows], self.df.iloc[rows]
        out = []
        for ch in self.channels:
            s, c = d[f"{ch}_spend"].sum(), dec[ch].sum()
            out.append({"channel": ch, "spend": s, "incremental_revenue": c,
                        "roi": c / s if s else np.nan})
        t = pd.DataFrame(out)
        t["share_of_spend"] = t.spend / t.spend.sum()
        t["share_of_effect"] = t.incremental_revenue / t.incremental_revenue.sum()
        return t

    def channel_params(self) -> dict:
        nb = self.n_base
        return {
            ch: {"decay": float(self.theta[3 * i]), "k": float(self.theta[3 * i + 1]),
                 "a": float(self.theta[3 * i + 2]), "scale": float(self.scales[ch]),
                 "coef": float(self.beta[nb + i]),
                 "avg_weekly_spend": float(self.df[f"{ch}_spend"].mean())}
            for i, ch in enumerate(self.channels)
        }


# ----------------------------------------------------------------------------
# Response curves and budget optimization
# ----------------------------------------------------------------------------
def weekly_response(spend, p: dict):
    """Steady-state weekly incremental revenue for a constant weekly spend."""
    x = np.asarray(spend, float) / p["scale"] / (1 - p["decay"])
    return p["coef"] * hill(x, p["k"], p["a"])


def optimize_budget(params: dict, total_budget: float, lower: float = 0.5, upper: float = 1.5):
    """Maximize media-driven revenue for a fixed total weekly budget.

    Each channel may move between lower*current and upper*current spend.
    Returns ({channel: optimized weekly spend}, optimized weekly media revenue).
    """
    chs = list(params)
    cur = np.array([params[c]["avg_weekly_spend"] for c in chs])
    ref = sum(float(weekly_response(c, params[ch])) for c, ch in zip(cur, chs)) or 1.0

    def obj(r):
        return -sum(float(weekly_response(ri * ci, params[ch]))
                    for ri, ci, ch in zip(r, cur, chs)) / ref

    cons = {"type": "eq", "fun": lambda r: (r * cur).sum() / total_budget - 1}
    r0 = np.full(len(chs), total_budget / cur.sum())
    res = minimize(obj, r0, method="SLSQP", bounds=[(lower, upper)] * len(chs),
                   constraints=[cons])
    return dict(zip(chs, res.x * cur)), -res.fun * ref


# ----------------------------------------------------------------------------
# Plot helper
# ----------------------------------------------------------------------------
def save_figure(path, fig=None, dpi: int = 120):
    """Save a figure; display it in notebooks, close it in scripts (so `python train.py` never blocks)."""
    fig = fig or plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    if "inline" in matplotlib.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)
