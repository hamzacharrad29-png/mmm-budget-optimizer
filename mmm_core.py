"""Lightweight Marketing Mix Model, at national or regional level.

Model (per region g and week t, everything per capita):

    revenue_gt / pop_g = alpha_g + base_t . gamma + controls_gt . delta
                         + sum_c beta_c * Hill(Adstock(spend_gct / pop_g / scale_c))

- alpha_g: one intercept per region (fixed effects), gamma: trend + yearly seasonality,
  delta: control variables, beta_c >= 0: media effects.
- Adstock decay and Hill (k, a) are shared by all regions and tuned with differential
  evolution; the linear coefficients are solved by bounded, population-weighted least squares.
- The national model is the same code with the 40 regions summed into one.
"""
from __future__ import annotations

from dataclasses import dataclass
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
LEVELS = ("national", "regional")


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


@dataclass
class Panel:
    """Weekly data as arrays: G regions x T weeks (G = 1 for the national level)."""
    geos: list
    times: pd.DatetimeIndex
    revenue: np.ndarray    # G x T
    spend: np.ndarray      # C x G x T
    controls: np.ndarray   # K x G x T
    population: np.ndarray  # G

    @property
    def shape(self):
        return len(self.geos), len(self.times)


def load_panel(level: str = "regional", path: Path = DATA_PATH) -> Panel:
    """Load the data at regional level (40 regions) or national level (regions summed)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    df = load_geo_data(path)
    df["revenue"] = df["conversions"] * df["revenue_per_conversion"]
    geos = sorted(df["geo"].unique(), key=lambda g: (len(str(g)), str(g)))  # Geo0, Geo1, ..., Geo39
    times = np.sort(df["time"].unique())

    def pivot(col):
        return df.pivot(index="geo", columns="time", values=col).loc[geos, times].to_numpy(float)

    revenue = pivot("revenue")
    spend = np.stack([pivot(f"{c}_spend") for c in CHANNELS])
    controls = np.stack([pivot(c) for c in CONTROLS])
    population = df.groupby("geo")["population"].first().loc[geos].to_numpy(float)
    if level == "national":
        revenue = revenue.sum(0, keepdims=True)
        spend = spend.sum(1, keepdims=True)
        controls = controls.mean(1, keepdims=True)
        population = population.sum(keepdims=True)
        geos = ["National"]
    return Panel(list(geos), pd.DatetimeIndex(times), revenue, spend, controls, population)


# ----------------------------------------------------------------------------
# Media transformations
# ----------------------------------------------------------------------------
def geometric_adstock(x, decay: float) -> np.ndarray:
    """Carry-over effect along the last axis: a_t = x_t + decay * a_{t-1}."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    carry = np.zeros(x.shape[:-1])
    for t in range(x.shape[-1]):
        carry = x[..., t] + decay * carry
        out[..., t] = carry
    return out


def hill(x, k: float, a: float):
    """Diminishing returns. k = half-saturation point, a = shape."""
    x = np.clip(x, 0, None)
    return x**a / (x**a + k**a)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class MMM:
    """Adstock + Hill regression with region fixed effects and non-negative media effects."""

    # Search ranges per channel: adstock decay, Hill half-saturation k (relative to the
    # maximum weekly spend), Hill shape a. k >= 0.3 and a <= 2 rule out near step-function
    # curves, which fit noise and make ROI estimates unstable.
    BOUNDS = [(0.0, 0.9), (0.3, 3.0), (0.5, 2.0)]

    def __init__(self, channels=CHANNELS):
        self.channels = list(channels)

    # -- design ----------------------------------------------------------------
    def _base(self, fit_end: int) -> np.ndarray:
        """Trend, yearly seasonality (Fourier) and standardized controls: P x G x T."""
        G, T = self.panel.shape
        t = np.arange(T)
        feats = [t / T] + [f(2 * np.pi * o * t / 52.18) for o in (1, 2) for f in (np.sin, np.cos)]
        base = [np.broadcast_to(f, (G, T)) for f in feats]
        for v in self.panel.controls:
            sd = v[:, :fit_end].std()
            base.append((v - v[:, :fit_end].mean()) / sd if sd > 0 else np.zeros((G, T)))
        self.base_names = ["trend", "sin1", "cos1", "sin2", "cos2", *CONTROLS]
        return np.stack(base)

    def _media(self, theta) -> np.ndarray:
        """Transformed media: C x G x T."""
        return np.stack([
            hill(geometric_adstock(self.x[i] / self.scales[i], theta[3 * i]),
                 theta[3 * i + 1], theta[3 * i + 2])
            for i in range(len(self.channels))])

    def _solve(self, M, end):
        """Population-weighted least squares on [0, end) with region fixed effects
        (removed by demeaning within each region) and media coefficients >= 0."""
        F = np.concatenate([self.B, M])[:, :, :end]
        yf = self.y[:, :end]
        Fd, yd = F - F.mean(2, keepdims=True), yf - yf.mean(1, keepdims=True)
        w = np.sqrt(self.panel.population)[:, None]
        A = (Fd * w).reshape(len(F), -1).T
        lb = np.r_[np.full(len(self.B), -np.inf), np.zeros(len(M))]
        coef = lsq_linear(A, (yd * w).ravel(), bounds=(lb, np.inf)).x
        alpha = yf.mean(1) - np.einsum("k,kg->g", coef, F.mean(2))
        return coef, alpha

    def _predict(self, M, coef, alpha) -> np.ndarray:
        F = np.concatenate([self.B, M])
        return (alpha[:, None] + np.einsum("k,kgt->gt", coef, F)) * self.panel.population[:, None]

    # -- public API ------------------------------------------------------------
    def fit(self, panel: Panel, fit_end: int, maxiter: int = 60, seed: int = 42):
        """Fit on weeks [0, fit_end). Later weeks are a hold-out test set, never used for fitting.

        The adstock / Hill parameters minimize the population-weighted in-sample error
        (differential evolution); for each candidate, the linear part is solved exactly.
        """
        self.panel, self.fit_end = panel, fit_end
        pop = panel.population
        self.y = panel.revenue / pop[:, None]
        self.x = panel.spend / pop[None, :, None]
        # Per-capita spend scaled by its maximum over the fitting window
        self.scales = np.array([self.x[i][:, :fit_end].max() or 1.0
                                for i in range(len(self.channels))])
        self.B = self._base(fit_end)
        Y = panel.revenue[:, :fit_end]

        def objective(theta):
            M = self._media(theta)
            P = self._predict(M, *self._solve(M, fit_end))[:, :fit_end]
            return np.sqrt(np.mean((Y - P) ** 2)) / Y.std()

        res = differential_evolution(objective, self.BOUNDS * len(self.channels),
                                     maxiter=maxiter, seed=seed, tol=1e-8, polish=False)
        self.theta = res.x
        self.M = self._media(self.theta)
        self.coef, self.alpha = self._solve(self.M, fit_end)
        return self

    @property
    def media_coef(self) -> np.ndarray:
        return self.coef[len(self.B):]

    def predict(self) -> np.ndarray:
        """Predicted revenue, G x T."""
        return self._predict(self.M, self.coef, self.alpha)

    def contributions(self) -> np.ndarray:
        """Incremental revenue per channel, C x G x T."""
        return self.media_coef[:, None, None] * self.M * self.panel.population[None, :, None]

    def decompose(self) -> pd.DataFrame:
        """Weekly revenue (summed over regions) split into baseline and channel contributions."""
        contrib = self.contributions().sum(1)
        dec = pd.DataFrame(contrib.T, columns=self.channels)
        dec.insert(0, "baseline", self.predict().sum(0) - contrib.sum(0))
        dec.insert(0, "time", self.panel.times)
        return dec

    def roi_table(self) -> pd.DataFrame:
        spend = self.panel.spend.sum((1, 2))
        inc = self.contributions().sum((1, 2))
        t = pd.DataFrame({"channel": self.channels, "spend": spend,
                          "incremental_revenue": inc, "roi": inc / spend})
        t["share_of_spend"] = t.spend / t.spend.sum()
        t["share_of_effect"] = t.incremental_revenue / t.incremental_revenue.sum()
        return t

    def fit_quality(self) -> dict:
        """R2 and MAPE on the fitting window and the test window, national totals and per region."""
        Y, P = self.panel.revenue, self.predict()
        windows = {"fit": slice(0, self.fit_end), "test": slice(self.fit_end, None)}
        out = {"national_total": {k: _metrics(Y.sum(0)[s], P.sum(0)[s]) for k, s in windows.items()}}
        if Y.shape[0] > 1:
            out["per_region"] = {k: _metrics(Y[:, s].ravel(), P[:, s].ravel()) for k, s in windows.items()}
        return out

    def channel_params(self) -> dict:
        """Everything the response curves and the budget optimizer need, per channel."""
        pop = self.panel.population
        out = {}
        for i, ch in enumerate(self.channels):
            s = self.panel.spend[i]
            out[ch] = {"decay": float(self.theta[3 * i]), "k": float(self.theta[3 * i + 1]),
                       "a": float(self.theta[3 * i + 2]), "scale": float(self.scales[i]),
                       "coef": float(self.media_coef[i]),
                       "avg_weekly_spend": float(s.sum(0).mean()),
                       "region_share": (s.sum(1) / s.sum()).tolist(),
                       "region_population": pop.tolist()}
        return out


def _metrics(actual, predicted) -> dict:
    r2 = 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    mape = np.mean(np.abs((actual - predicted) / actual))
    return {"R2": round(float(r2), 4), "MAPE": round(float(mape), 4)}


# ----------------------------------------------------------------------------
# Response curves and budget optimization
# ----------------------------------------------------------------------------
def weekly_response(spend, p: dict):
    """Steady-state weekly incremental revenue for a constant total weekly spend.

    The spend is split across regions in their historical proportions.
    """
    spend = np.asarray(spend, float)
    share, pop = np.asarray(p["region_share"]), np.asarray(p["region_population"])
    x = spend[..., None] * share / pop / p["scale"] / (1 - p["decay"])
    return (p["coef"] * hill(x, p["k"], p["a"]) * pop).sum(-1)


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
    """Save a figure; display it in notebooks, close it in scripts (so scripts never block)."""
    fig = fig or plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    if "inline" in matplotlib.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)
