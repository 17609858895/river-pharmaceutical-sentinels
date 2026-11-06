"""Threshold-agnostic risk inference: one censored structured model vs threshold-specific classifiers.

Endpoints: any API above c x PNEC_lowest for c in {0.1, 0.3, 1, 3, 10}, and any antibiotic above PNEC-AMR.
  SENTRY            one panel-specific censored AFT + copula model; new thresholds only change the aggregation
  Direct-transfer   LightGBM trained on the c = 1 (primary screening benchmark) labels, scored on every endpoint
  Direct-retrained  LightGBM retrained on each endpoint's own labels (uses those labels; strong reference)
Same outer folds as 10_sentinel_v3.py; panels: none, union heuristic k = 3 and k = 6.
Usage: python -u scripts/14_threshold_transfer.py SEED [FOLD]
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import sentry_lib as S

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT = S.ROOT / "results" / "threshold_transfer"
OUT.mkdir(parents=True, exist_ok=True)
d = S.load()
apis = d["apis"]; n, p = d["det"].shape
targets = [j for j in range(p) if d["det"][:, j].sum() >= 5]
KW = dict(S.LGB_KW, n_jobs=S.NTHREAD)
ENDPOINTS = {f"PNECx{c}": d["pnec"] * c for c in [0.1, 0.3, 1, 3, 10]}
ENDPOINTS["PNEC-AMR"] = d["pnec_amr"]
Y = {k: S.observed_endpoints(d, np.arange(n), v)[0] for k, v in ENDPOINTS.items()}

ug = np.unique(d["groups"]); perm = np.random.default_rng(SEED).permutation(len(ug))
gshuf = pd.Series(d["groups"]).map(dict(zip(ug, perm))).values
rows = []
for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(n), groups=gshuf)):
    if ONLY is not None and fold != ONLY:
        continue
    inner = list(GroupKFold(3).split(np.zeros(len(tr)), groups=gshuf[tr]))
    freq = d["det"][tr].mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        exc = np.nan_to_num((d["v_obs"][tr] / d["pnec"][None, :]) > 1).mean(0)
    pool = sorted({j for j in range(p) if freq[j] >= 0.05} & set(targets))
    fr = [j for j in np.argsort(-freq) if j in pool]; er = [j for j in np.argsort(-exc) if j in pool]
    union = []
    for a, b in zip(er, fr):
        for c in (a, b):
            if c not in union:
                union.append(c)
    for pname, pan in [("none", []), ("union_k3", union[:3]), ("union_k6", union[:6])]:
        st = S.structured_run(d, tr, te, inner, targets, pan, d["pnec"], "aft")
        Xd = S.direct_features(d, pan)
        base = lgb.LGBMClassifier(**KW).fit(Xd[tr], Y["PNECx1"][tr]).predict_proba(Xd[te])[:, 1]
        for ep, thr in ENDPOINTS.items():
            y = Y[ep][te]
            if y.sum() < 3 or y.sum() > len(y) - 3:
                continue
            meas = S.measured_part(d, te, pan, thr)
            ps = S.site_endpoints_dep(d, te, targets, pan, st["mu_te"], st["sig"], thr, st["rho"], n_draw=20)[0]
            ytr = Y[ep][tr]
            pr = lgb.LGBMClassifier(**KW).fit(Xd[tr], ytr).predict_proba(Xd[te])[:, 1] if 3 <= ytr.sum() else base
            for m, q in [("SENTRY", ps), ("Direct-transfer", np.where(meas, 1.0, base)),
                         ("Direct-retrained", np.where(meas, 1.0, pr))]:
                rows.append(dict(seed=SEED, fold=fold, panel=pname, endpoint=ep, prevalence=y.mean(),
                                 train_pos=int(ytr.sum()), model=m, AUC=roc_auc_score(y, q),
                                 PRAUC=average_precision_score(y, q)))
        print(f"seed {SEED} fold {fold} {pname} done", flush=True)
    tag = f"seed{SEED}" + (f"_fold{ONLY}" if ONLY is not None else "")
    pd.DataFrame(rows).to_csv(OUT / f"transfer_{tag}.csv", index=False)
