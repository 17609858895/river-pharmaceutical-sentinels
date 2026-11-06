"""Current threshold-transfer, calibration and partial-panel plot builders."""
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
tt = pd.read_csv(R / "threshold_transfer" / "transfer_all.csv")
cal = pd.read_csv(R / "sentinel_v3" / "calibration_check.csv")
pp = pd.read_csv(R / "partial_panels" / "partial_panels_all.csv")
pp["ptype"] = np.where(pp.panel.str.startswith("random"), pp.panel.str.replace("random-k", "Random k="), pp.panel)




def p5b(ax):
    order = ["PNECx0.1", "PNECx0.3", "PNECx1", "PNECx3", "PNECx10", "PNEC-AMR"]
    lab = ["0.1×", "0.3×", "1×", "3×", "10×", "AMR"]
    sub = tt[tt.panel == "union_k6"].groupby(["endpoint", "model"]).PRAUC.mean().unstack().reindex(order)
    x = np.arange(len(order))
    for m, c, mk in [("SENTRY", pal[1], "o"), ("Direct-transfer", pal[0], "s"), ("Direct-retrained", pal[2], "^")]:
        ax.plot(x, sub[m], marker=mk, color=c, mec=INK, ms=8, lw=2, label=m)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    style_axis(ax, "Benchmark multiplier", "PR-AUC (k = 6)"); numeric_ticks(ax, x=False, n=5)
    ax.legend(fontsize=10, loc="lower left")


def p5c(ax):
    order = ["Direct raw", "Direct + Platt", "Direct + isotonic", "SENTRY"]
    g = cal.groupby("model").LogLoss.agg(["mean", "std"]).reindex(order)
    ax.bar(range(4), g["mean"], yerr=g["std"], color=[pal[0], pal[3], pal[4], pal[1]], edgecolor=INK, lw=0.8,
           capsize=3, width=0.62)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Direct\nraw", "Direct\n+Platt", "Direct\n+isotonic", "SENTRY"])
    style_axis(ax, None, "Exceedance log loss", tick_size=10.5); numeric_ticks(ax, x=False, n=4)




ORDER6 = ["Antibiotics (13)", "Wastewater markers", "CNS drugs", "High-volume drugs", "Random k=1", "Random k=3",
          "Random k=5", "Random k=8", "Random k=12"]


def p6a(ax):
    h = pp[pp.method == "SENTRY"].groupby("ptype").hidden_share.mean().reindex(ORDER6) * 100
    ax.barh(range(len(h)), h.values, color=pal[1], edgecolor=INK, lw=0.6)
    ax.set_yticks(range(len(h))); ax.set_yticklabels(h.index, fontsize=10.5); ax.invert_yaxis()
    style_axis(ax, "Exceeding sites missed (%)", None, tick_size=10.5); numeric_ticks(ax, y=False, n=5)


def p6b(ax):
    g = pp.groupby(["ptype", "method"]).AUC.mean().unstack().reindex(ORDER6)
    y = np.arange(len(g))
    for m, c, mk in [("Measured-only", pal[4], "D"), ("Direct-LGBM", pal[0], "s"), ("SENTRY", pal[1], "o")]:
        ax.scatter(g[m], y, color=c, edgecolor=INK, s=60, marker=mk, label=m, zorder=3)
    for i in y:
        ax.plot([g["Measured-only"].iloc[i], max(g["SENTRY"].iloc[i], g["Direct-LGBM"].iloc[i])], [i, i], color="#DDDDDD",
                zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=10.5); ax.invert_yaxis()
    style_axis(ax, "Exceedance ROC AUC", None, tick_size=10.5); numeric_ticks(ax, y=False, n=5)
    ax.legend(fontsize=10, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, columnspacing=.8, handletextpad=.3)



