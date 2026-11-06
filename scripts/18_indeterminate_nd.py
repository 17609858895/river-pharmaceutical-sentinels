"""Risk-indeterminate non-detects: APIs whose LOD exceeds the PNEC (loratadine; itraconazole has 1 detection).

For a non-detect, C is in (0, LOD); with LOD > PNEC the probability of an unseen exceedance is
    P(PNEC < C < LOD | C < LOD, x, o^S) = [Phi(z_LOD) - Phi(z_PNEC)] / Phi(z_LOD),  z_t = (ln t - mu) / sigma,
from the censored AFT predictive distribution (out-of-fold, leave-campaign-out, seed 0).
Reports expected hidden exceedances and how site-level mixture risk changes when they are counted.
Usage: python -u scripts/18_indeterminate_nd.py
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

import sentry_lib as S

OUT = S.ROOT / "results" / "indeterminate"
OUT.mkdir(parents=True, exist_ok=True)
d = S.load()
apis = d["apis"]; n, p = d["det"].shape
targets = [j for j in range(p) if d["det"][:, j].sum() >= 5]
pnec = d["pnec"]
indet = [j for j in range(p) if np.isfinite(pnec[j]) and d["lod"][j] > pnec[j]]
print("LOD > PNEC:", [(apis[j], d["lod"][j], pnec[j], int(d["det"][:, j].sum())) for j in indet])
y_risk = S.observed_endpoints(d, np.arange(n), pnec)[0]
ug = np.unique(d["groups"]); perm = np.random.default_rng(0).permutation(len(ug))
gshuf = pd.Series(d["groups"]).map(dict(zip(ug, perm))).values
rows = []
for pname in ["none", "union_k6"]:
    p_hidden = np.full((n, len(indet)), np.nan)
    for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(n), groups=gshuf)):
        inner = list(GroupKFold(3).split(np.zeros(len(tr)), groups=gshuf[tr]))
        pan = []
        if pname == "union_k6":
            freq = d["det"][tr].mean(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                exc = np.nan_to_num((d["v_obs"][tr] / pnec[None, :]) > 1).mean(0)
            pool = sorted({j for j in range(p) if freq[j] >= 0.05} & set(targets))
            fr = [j for j in np.argsort(-freq) if j in pool]; er = [j for j in np.argsort(-exc) if j in pool]
            for a, b in zip(er, fr):
                for c in (a, b):
                    if c not in pan:
                        pan.append(c)
            pan = pan[:6]
        st = S.structured_run(d, tr, te, inner, targets, pan, pnec, "aft")
        for i, j in enumerate(indet):
            if j not in targets:
                continue
            t = targets.index(j)
            mu, sg = st["mu_te"][:, t], st["sig"][t]
            zl = (np.log(d["lod"][j]) - mu) / sg; zp = (np.log(pnec[j]) - mu) / sg
            ph = (norm.cdf(zl) - norm.cdf(zp)) / np.clip(norm.cdf(zl), 1e-12, None)
            nd = (d["status"][te, j] == "ND")
            p_hidden[te[nd], i] = ph[nd]
        print(f"{pname} fold {fold} done", flush=True)
    for i, j in enumerate(indet):
        ph = p_hidden[:, i]; nd = ~np.isnan(ph)
        if not nd.any():
            rows.append(dict(panel=pname, api=apis[j], n_nd=int((d["status"][:, j] == "ND").sum()), modelled=False))
            continue
        safe_sites = nd & (y_risk == 0)
        rows.append(dict(panel=pname, api=apis[j], modelled=True, n_nd=int(nd.sum()),
                         n_detected=int(d["det"][:, j].sum()), expected_hidden_exceed=float(np.nansum(ph)),
                         mean_p=float(np.nanmean(ph)), n_p_gt_0_5=int((ph > 0.5).sum()),
                         n_safe_sites_p_gt_0_5=int((ph[safe_sites] > 0.5).sum()),
                         expected_new_risk_sites=float(np.nansum(ph[safe_sites])),
                         share_of_safe_sites=float(np.nansum(ph[safe_sites]) / (y_risk == 0).sum())))
        pd.DataFrame({"site_id": d["F"].site_id, "campaign": d["F"].campaign, "continent": d["F"].continent,
                      "p_hidden": ph, "observed_risk": y_risk}).to_csv(OUT / f"p_hidden_{apis[j]}_{pname}.csv", index=False)
res = pd.DataFrame(rows)
res.to_csv(OUT / "indeterminate_summary.csv", index=False)
print(res.round(3).to_string())
