"""Final panel builders for benchmark, information-link and site-context figures."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "figlib"))
sys.path.insert(0, str(Path(__file__).parent))
from ml_publication_style import configure_publication_style, style_axis, numeric_ticks  # noqa: E402
from scientific_palettes import get_palette  # noqa: E402
import sentry_lib as S  # noqa: E402

configure_publication_style()
pal = get_palette("au_nimno_soft_8")
INK = "#333333"
R = S.ROOT / "results"


def load_bench(tag):
    f = R / "benchmark20" / f"bench20_{tag}_summary.csv"
    return pd.read_csv(f) if f.exists() else None


cov, rnd, chw = load_bench("cov"), load_bench("cov_randomsplit"), load_bench("cov_hw")


def bench_panel(ax, task, col, xlabel, lo, legend=True):
    a = cov[cov.task == task].set_index("model")[col]
    c = rnd[rnd.task == task].set_index("model")[col] if rnd is not None else None
    order = a.sort_values().index
    order = [m for m in order if a.get(m, -9) > lo - 1]
    y = np.arange(len(order))
    for i, m in enumerate(order):
        ax.plot([max(a[m], lo), c.get(m, a[m])], [i, i], color="#CCCCCC", lw=1.5, zorder=1)
    ax.scatter(np.clip(a[order], lo, None), y, s=42, color=pal[0], edgecolor=INK, lw=0.6, zorder=3,
               label="Covariates: held-out campaigns")
    if c is not None:
        ax.scatter(np.clip(c.reindex(order), lo, None), y, s=42, marker="D", color=pal[4], edgecolor=INK, lw=0.9,
                   zorder=3, label="Covariates: random rows")
    if chw is not None:
        h = chw[chw.task == task].set_index("model")[col]
        ax.scatter(np.clip(h.reindex(order), lo, None), y, s=30, marker="^", color=pal[2], edgecolor=INK, lw=0.6,
                   zorder=3, label="+ wastewater infrastructure")

    # A left-pointing marker explicitly flags values outside the displayed range.
    for values, colour in [(a.reindex(order), pal[0]), (c.reindex(order), pal[4]), (h.reindex(order), pal[2])]:
        clipped = np.asarray(values) < lo
        if clipped.any():
            ax.scatter(np.full(clipped.sum(), lo + .012 * (1 - lo)), y[clipped], marker="<", s=85,
                       color=colour, edgecolor=INK, linewidth=.6, zorder=8, clip_on=False)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlim(lo, 1.0)
    style_axis(ax, xlabel, None, tick_size=10.5)
    numeric_ticks(ax, y=False, n=5)
    if legend:
        ax.legend(fontsize=10.5, loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=2, columnspacing=1.0, handletextpad=0.3)


net = pd.read_csv(R / "interpret" / "sentinel_information_network.csv", index_col=0)
M = net.copy()
for t in M.index:
    if t in M.columns:
        M.loc[t, t] = np.nan
spec = M.sub(M.mean(1), axis=0).sub(M.mean(0), axis=1) + np.nanmean(M.values)
hubs = M.mean().sort_values(ascending=False)


def net_panel(ax):
    src = list(hubs.index[:12])
    tgt = list(M[src].mean(1).sort_values(ascending=False).index[:16])
    Z = spec.loc[tgt, src].values
    vmax = np.nanpercentile(np.abs(Z), 98)
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list("pal_div", [pal[0], "#FFFFFF", pal[1]])  # zero-centred, palette ends
    im = ax.imshow(Z, cmap=div, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(src))); ax.set_xticklabels(src, rotation=55, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(tgt))); ax.set_yticklabels(tgt, fontsize=9.5)
    style_axis(ax, "Observed API (source)", "Predicted API (target)", tick_size=9.5, label_size=13)
    for lab in ax.get_xticklabels():
        lab.set_rotation(55); lab.set_ha("right")
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
    cb.set_label("Centred mean |SHAP|", fontsize=10, fontweight="normal"); cb.ax.tick_params(labelsize=9)


def hub_panel(ax):
    h = hubs.iloc[:12][::-1]
    ax.barh(range(len(h)), h.values, color=pal[0], edgecolor=INK, lw=0.6)
    ax.set_yticks(range(len(h))); ax.set_yticklabels(h.index, fontsize=10.5)
    style_axis(ax, "Mean |SHAP| to other APIs", None, tick_size=10.5); numeric_ticks(ax, y=False, n=4)


dml = pd.read_csv(R / "causal" / "dml_wastewater_exposure.csv")


def dml_panel(ax):
    w = dml[dml.design == "within-campaign"].copy()
    w = w[w.outcome.str.startswith("log10")]
    w = w.iloc[::-1]
    y = np.arange(len(w))
    ax.errorbar(w.theta, y, xerr=1.96 * w.se, fmt="o", color=pal[1], mec=INK, ms=8, capsize=3, lw=1.5)
    pl = dml[dml.design.str.startswith("placebo")]
    # Permutations were run only for cumulative concentration, not every outcome.
    cumulative_row = y[w.outcome.str.contains("cumulative", case=False).to_numpy()][0]
    ylim_before_band = ax.get_ylim()
    ax.fill_betweenx([cumulative_row - .36, cumulative_row + .36],
                     pl.theta.quantile(.025), pl.theta.quantile(.975), color=pal[4], zorder=0)
    ax.set_ylim(ylim_before_band)
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([o.replace("log10 ", "") for o in w.outcome], fontsize=10.5)
    style_axis(ax, "Adjusted difference (Δ log$_{10}$)", None, tick_size=10.5); numeric_ticks(ax, y=False, n=5)

