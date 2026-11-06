"""Final Fig. 4 selection-curve and Pareto panel builders (Direct-LGBM)."""
from __future__ import annotations

import glob
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
res = pd.read_csv(R / "sentinel_v3" / "v3_plus_extra_all.csv")
res = res[res.model == "Direct-LGBM"].copy()
res["strategy"] = res.strategy.str.replace(r"random\d", "random", regex=True)
g = res.groupby(["strategy", "k"])[["AUC", "R2_logcum"]]
summ = g.mean().join(g.std(), rsuffix="_sd").reset_index()
k0 = summ[summ.strategy == "none"]

STYLE = {  # strategy: (label, colour, marker, linestyle)
    "exceed_rank": ("Exceedance rank", pal[1], "o", "-"),
    "load_greedy": ("Concentration-greedy", pal[0], "s", "-"),
    "joint0.5": ("Joint greedy", pal[2], "^", "-"),
    "union_heur": ("Alternating rule", pal[5], "D", "--"),
    "random": ("Random", pal[4], "o", ":"),
}


def curve(ax, metric, ylabel, legend=False):
    for st, (lab, c, mk, ls) in STYLE.items():
        sub = pd.concat([k0, summ[summ.strategy == st]]).sort_values("k")
        ax.errorbar(sub.k, sub[metric], yerr=sub[metric + "_sd"], color=c, marker=mk, ms=7, lw=2.2, ls=ls,
                    capsize=2.5, mec=INK, mew=0.8, ecolor=c, label=lab)
    style_axis(ax, "Measured sentinels (k)", ylabel); ax.set_xticks(range(0, 7)); numeric_ticks(ax, x=False, n=5)
    if legend:
        ax.legend(fontsize=9.5, loc="lower right", handlelength=2.6)


PARETO = {"exceed_rank": ("Exceedance rank", pal[1]), "risk_greedy": ("Risk-greedy", pal[1]),
          "load_greedy": ("Concentration-greedy", pal[0]), "frequency": ("Detection rank", pal[0]),
          "joint0.25": ("Joint λ=0.25", pal[2]), "joint0.5": ("Joint λ=0.5", pal[2]), "joint0.75": ("Joint λ=0.75", pal[2]),
          "union_heur": ("Alternating rule", pal[5]), "expert": ("Expert panel", pal[7]), "random": ("Random", pal[4])}
OFFS = {"exceed_rank": (6, 24, "left"), "risk_greedy": (16, -26, "left"), "load_greedy": (8, -34, "center"),
        "frequency": (-48, -36, "right"), "joint0.25": (40, -30, "left"), "joint0.5": (-24, -40, "right"),
        "joint0.75": (-26, 28, "right"), "union_heur": (18, 30, "left"), "expert": (-36, -24, "right"),
        "random": (14, 14, "left")}


def pareto(ax, k=6):
    for st, (lab, c) in PARETO.items():
        sub = summ[(summ.strategy == st) & (summ.k == k)]
        if sub.empty:
            continue
        x, y = sub.R2_logcum.iloc[0], sub.AUC.iloc[0]
        ax.errorbar(x, y, xerr=sub.R2_logcum_sd.iloc[0], yerr=sub.AUC_sd.iloc[0], fmt="o", color=c, mec=INK, mew=0.8,
                    ms=9, ecolor=c, elinewidth=1.2, capsize=2)
        dx, dy, ha = OFFS[st]
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9.5, fontweight="normal", ha=ha,
                    va="center", arrowprops=dict(arrowstyle="-", lw=0.7, color=INK, shrinkA=0, shrinkB=5))
    style_axis(ax, f"Concentration R$^2$ (k = {k})", f"Exceedance ROC AUC (k = {k})"); numeric_ticks(ax, n=4)
    ax.set_xlim(.43, .96); ax.set_ylim(.87, 1.01)

