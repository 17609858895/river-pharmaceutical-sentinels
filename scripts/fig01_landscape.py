"""Final Fig. 1 panel builders: survey, censoring, shares and income groups."""
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
PAL = get_palette("au_nimno_soft_8")
BLUE, RED, GREEN, LBLUE = PAL[0], PAL[1], PAL[2], PAL[3]
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
SEQ = LinearSegmentedColormap.from_list("pal_seq", [PAL[4], PAL[0], PAL[1]])  # palette-only sequential scale
INK = "#222222"
FIG = S.ROOT / "figures" / "Fig01_landscape"
(FIG / "subfigures").mkdir(parents=True, exist_ok=True)

d = S.load()
F = d["F"]; apis = d["apis"]; cp = d["cp"]
y_risk, y_cum, y_rq = S.observed_endpoints(d, np.arange(len(F)), d["pnec"])
det_freq = d["det"].mean(0)
with np.errstate(invalid="ignore", divide="ignore"):
    rq = np.nan_to_num(d["v_obs"] / d["pnec"][None, :])
n_exc = (rq > 1).sum(0)
mass_share = d["v_obs"].sum(0) / d["v_obs"].sum()
exc_share = n_exc / n_exc.sum()


try:  # Natural Earth 1:110m countries (public domain), downloaded with user approval
    import geopandas as gpd
    WORLD = gpd.read_file(S.ROOT / "external_data" / "naturalearth" / "ne_110m_admin_0_countries.shp")
except Exception:  # basemap is optional
    WORLD = None


def panel_a(ax):
    if WORLD is not None:
        WORLD.plot(ax=ax, color="#F2F2F2", edgecolor="#BDBDBD", linewidth=0.4, zorder=0)
    order = np.argsort(y_cum)
    sc = ax.scatter(F.lon.values[order], F.lat.values[order], c=y_cum[order], cmap=SEQ, vmin=0, vmax=5,
                    s=16, edgecolor=INK, linewidth=0.25)
    risky = y_risk == 1
    ax.scatter(F.lon[risky], F.lat[risky], s=40, facecolor="none", edgecolor=INK, linewidth=0.9)
    ax.set_xlim(-180, 180); ax.set_ylim(-66, 78); ax.set_aspect("auto")
    style_axis(ax, "Longitude (°)", "Latitude (°)")
    ax.set_xticks([-120, -60, 0, 60, 120]); ax.set_yticks([-60, -30, 0, 30, 60])
    cb = plt.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.24, aspect=40, extend="max")
    cb.set_label("log$_{10}$(1 + concentration [ng L$^{-1}$])", fontsize=11, fontweight="normal")
    cb.ax.tick_params(labelsize=11)
    ax.scatter([], [], s=40, facecolor="none", edgecolor=INK, linewidth=0.9, label="≥1 benchmark exceedance")
    ax.legend(loc="lower left", fontsize=10.5, handletextpad=0.3)


def panel_b(ax):
    ok = ~np.isnan(d["pnec"]) & (det_freq > 0)
    ratio = np.log10(d["lod"][ok] / d["pnec"][ok])
    x = det_freq[ok] * 100
    sz = 18 + 2.2 * n_exc[ok]
    col = np.where(ratio > 0, RED, BLUE)
    ax.scatter(x, ratio, s=sz, c=col, edgecolor=INK, linewidth=0.5, alpha=0.9)
    ax.axhline(0, color=INK, lw=1.0, ls="--")
    names = np.array(apis)[ok]
    offs = {"Propranolol": (-70, 18), "Ciprofloxacin": (8, -2), "Fluconazole": (-70, -16), "Metronidazole": (6, -14),
            "Clarithromycin": (8, -2), "Enrofloxacin": (-70, 8), "Sulfamethoxazole": (-85, -12), "Trimethoprim": (-80, -4)}
    for xi, yi, nm, ne in zip(x, ratio, names, n_exc[ok]):
        if nm in {"Loratadine", "Itraconazole", "Propranolol", "Ciprofloxacin", "Sulfamethoxazole"}:
            position = (.18, -3.2) if nm == "Sulfamethoxazole" else offs.get(nm, (5, 3))
            coordinates = "data" if nm == "Sulfamethoxazole" else "offset points"
            ax.annotate(nm, (xi, yi), xytext=position, textcoords=coordinates, fontsize=9.5,
                        fontweight="normal", arrowprops=dict(arrowstyle="-", lw=.55, color=INK, shrinkA=2, shrinkB=4))
    ax.set_xscale("log")
    style_axis(ax, "Detection frequency (%)", "log$_{10}$(LOD / benchmark)")
    ax.set_xticks([0.1, 1, 10, 50]); ax.set_xticklabels(["0.1", "1", "10", "50"])
    for lab in ax.get_xticklabels():
        lab.set_fontweight("normal")


def panel_c(ax):
    top = np.argsort(-(mass_share + exc_share))[:14]
    top = top[np.argsort(mass_share[top] - exc_share[top])]
    yy = np.arange(len(top))
    ax.barh(yy - 0.2, mass_share[top] * 100, height=0.38, color=BLUE, edgecolor=INK, linewidth=0.5, label="Concentration share")
    ax.barh(yy + 0.2, exc_share[top] * 100, height=0.38, color=RED, edgecolor=INK, linewidth=0.5, label="Exceedance share")
    ax.set_yticks(yy); ax.set_yticklabels([apis[j] for j in top])
    style_axis(ax, "Share of total (%)", None, tick_size=11)
    numeric_ticks(ax, y=False, n=4)
    ax.legend(loc="center right", fontsize=10.5)


def panel_d(ax):
    inc = d["X"].income_ord.values
    labs = ["Low", "Lower-\nmiddle", "Upper-\nmiddle", "High"]
    share = [y_risk[inc == k].mean() * 100 for k in range(4)]
    nn = [(inc == k).sum() for k in range(4)]
    ax.bar(range(4), share, color=[RED, RED, LBLUE, BLUE], edgecolor=INK, linewidth=0.8, width=0.62)
    for i, (s_, n_) in enumerate(zip(share, nn)):
        ax.text(i, s_ + 1.2, f"n={n_}", ha="center", fontsize=10.5, fontweight="normal")
    ax.text(.98, .98, f"n = {sum(nn):,}", transform=ax.transAxes, ha="right", va="top", fontsize=11, fontweight="normal")
    ax.set_xticks(range(4)); ax.set_xticklabels(labs)
    style_axis(ax, "Income group", "Sites with exceedance (%)", tick_size=11.5)
    numeric_ticks(ax, x=False, n=5)


PANELS = [("a", panel_a, (7.2, 4.4)), ("b", panel_b, (5.4, 4.4)), ("c", panel_c, (5.0, 5.2)), ("d", panel_d, (4.6, 4.4))]

if __name__ == "__main__":
    from figures_publication import fig1
    fig1()
