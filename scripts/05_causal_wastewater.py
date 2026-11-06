"""Recorded wastewater association estimated by campaign-grouped DML.

Exposure T (direction-aware text rule on unedited collector notes):
    T=1  site recorded downstream of / receiving WWTP effluent, sewage or wastewater discharge
    T=0  all other sites (including sites explicitly upstream of a WWTP)
Design: partially linear DML (Chernozhukov et al. 2018) with campaign-grouped cross-fitting and
campaign-demeaned outcome/treatment (within-campaign adjusted contrast).
Post-treatment variables (EC, TDS, pH, temperature) are excluded from the adjustment set.
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

import re

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GroupKFold

import sentry_lib as S

OUT = S.ROOT / "results" / "causal"
OUT.mkdir(parents=True, exist_ok=True)
d = S.load()
sites = pd.read_csv(S.D / "sites.csv")
F = d["F"]
txt = sites.set_index("site_id").reindex(F.site_id).notes.fillna("").str.lower().values

WW = r"(wwtp|stp\b|sewage|sewer|wastewater|waste water|effluent|outfall|treatment (plant|works|fac)|discharge|outlet)"
UP = re.compile(r"(upstream|before|prior to|above)\W+(?:\S+\W+){0,6}?" + WW)
DOWN = re.compile(r"(downstream|after|below|receiv\w*|inflows?|joined|directly)\W+(?:\S+\W+){0,6}?" + WW)
DIRECT = re.compile(r"raw sewage|smell\w* of sewage|receives? waste ?waters?|sewage (water|discharge|pipe)|"
                    r"direct (wastewater|effluent)|effluent (outfall|channel|stream)|waste water disposal|"
                    r"untreated (discharge|sewage|wastewater)")


def exposure(t: str) -> int:
    """A downstream/receiving or direct-discharge statement marks exposure; an upstream-only
    statement (e.g. '500 m upstream from WWTP discharge') does not."""
    return int(bool(t) and (bool(DOWN.search(t)) or bool(DIRECT.search(t))))


T = np.array([exposure(t) for t in txt])
UPX = np.array([int(bool(UP.search(t)) and not exposure(t)) for t in txt])
print("exposed sites:", T.sum(), "explicit-upstream controls:", UPX.sum(), "campaigns with exposure:",
      len(np.unique(F.campaign_id[T == 1])))

y_risk, y_cum, y_rq = S.observed_endpoints(d, np.arange(len(F)), d["pnec"])
n_det = d["det"].sum(1)
post = {"temp_C", "pH", "log_flow_m3s", "log_TDS_ppm", "log_EC_uScm", "miss_temp_C", "miss_pH", "miss_log_flow_m3s",
        "miss_log_TDS_ppm", "miss_log_EC_uScm", "kw_wastewater", "kw_downstream", "kw_odour"}
Xc = d["X"][[c for c in d["X"].columns if c not in post]]
g = d["groups"]


def demean(v, groups):
    return v - pd.Series(v).groupby(groups).transform("mean").values


def dml_plr(y, t, X, groups, within=True, seeds=(0, 1, 2)):
    thetas, ses = [], []
    for sd in seeds:
        ug = np.unique(groups); perm = np.random.default_rng(sd).permutation(len(ug))
        gg = pd.Series(groups).map(dict(zip(ug, perm))).values
        yr, tr_ = np.zeros(len(y)), np.zeros(len(y))
        for a, b in GroupKFold(5).split(X, groups=gg):
            my = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=3, random_state=sd)
            yr[b] = y[b] - my.fit(X.iloc[a], y[a]).predict(X.iloc[b])
            mt = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=3, random_state=sd)
            tr_[b] = t[b] - np.clip(mt.fit(X.iloc[a], t[a]).predict_proba(X.iloc[b])[:, 1], 0.02, 0.98)
        if within:  # campaign fixed effects on the residualised scale
            yr, tr_ = demean(yr, groups), demean(tr_, groups)
        th = (tr_ * yr).sum() / (tr_ ** 2).sum()
        psi = tr_ * (yr - th * tr_)
        # campaign-clustered sandwich SE
        cl = pd.Series(psi).groupby(groups).sum().values
        se = np.sqrt((cl ** 2).sum()) / (tr_ ** 2).sum()
        thetas.append(th); ses.append(se)
    th = float(np.median(thetas)); se = float(np.median(ses))
    return th, se


rows = []
outcomes = {"log10 cumulative (ng/L)": y_cum, "log10 sum RQ": y_rq, "n APIs detected": n_det.astype(float),
            "any RQ>1": y_risk.astype(float)}
for cls in ["Antibiotic", "Anticonvulsants", "Antihyperglycemic", "Analgesic", "Beta-blocker", "Antidepressant"]:
    js = [j for j, a in enumerate(d["apis"]) if d["cp"].loc[a, "therapeutic_class"] == cls]
    outcomes[f"log10 {cls}"] = np.log10(d["v_obs"][:, js].sum(1) + 1)
for name, y in outcomes.items():
    for within in [False, True]:
        th, se = dml_plr(y, T, Xc, g, within=within)
        rows.append(dict(outcome=name, design="within-campaign" if within else "pooled", theta=th, se=se,
                         lo=th - 1.96 * se, hi=th + 1.96 * se, z=th / se))
# placebo: shuffled exposure within campaign
rng = np.random.default_rng(7)
for b in range(20):
    Tp = pd.Series(T).groupby(g).transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index)).values
    th, se = dml_plr(y_cum, Tp, Xc, g, within=True, seeds=(0,))
    rows.append(dict(outcome="log10 cumulative (ng/L)", design=f"placebo{b}", theta=th, se=se, lo=th - 1.96 * se,
                     hi=th + 1.96 * se, z=th / se))
# heterogeneity by income group (within-campaign DML on subsets)
inc = F.merge(pd.read_csv(S.D / "site_features.csv")[["site_id"]], on="site_id").pipe(lambda x: d["X"].income_ord.values)
for lab, msk in [("Low/lower-middle income", inc <= 1), ("Upper-middle income", inc == 2), ("High income", inc == 3)]:
    if T[msk].sum() >= 10:
        th, se = dml_plr(y_cum[msk], T[msk], Xc[msk].reset_index(drop=True), g[msk], within=True)
        rows.append(dict(outcome="log10 cumulative (ng/L)", design=f"within | {lab}", theta=th, se=se,
                         lo=th - 1.96 * se, hi=th + 1.96 * se, z=th / se, n_exposed=int(T[msk].sum())))
res = pd.DataFrame(rows)
res.to_csv(OUT / "dml_wastewater_exposure.csv", index=False)
pd.DataFrame({"site_id": F.site_id, "T_wastewater": T, "explicit_upstream": UPX}).to_csv(OUT / "exposure_labels.csv", index=False)
print(res[~res.design.str.startswith("placebo")].round(3).to_string())
pl = res[res.design.str.startswith("placebo")]
print("placebo theta mean %.3f sd %.3f; share |z|>1.96: %.2f" % (pl.theta.mean(), pl.theta.std(), (pl.z.abs() > 1.96).mean()))
