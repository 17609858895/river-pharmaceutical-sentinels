"""Compare raw/calibrated Direct-LGBM probabilities with SENTRY.

Same outer folds/panels as 10_sentinel_v3.py + 12_extra_baselines.py. Direct-LGBM inner-fold OOF probabilities
are used to fit Platt (logistic on logit) and isotonic recalibration maps; the outer model's test probabilities
are recalibrated and scored. SENTRY numbers are read from the v3/extra result files for the same panels.
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

import glob
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import sentry_lib as S

d = S.load()
apis = d["apis"]; n, p = d["det"].shape
pnec = d["pnec"]
y_risk = S.observed_endpoints(d, np.arange(n), pnec)[0]
KW = dict(S.LGB_KW, n_jobs=S.NTHREAD)
R = S.ROOT / "results" / "sentinel_v3"
allr = pd.read_csv(R / "v3_plus_extra_all.csv")
rows = []
for seed in [0, 1]:
    ug = np.unique(d["groups"]); perm = np.random.default_rng(seed).permutation(len(ug))
    gshuf = pd.Series(d["groups"]).map(dict(zip(ug, perm))).values
    for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(n), groups=gshuf)):
        inner = list(GroupKFold(3).split(np.zeros(len(tr)), groups=gshuf[tr]))
        sub = allr[(allr.seed == seed) & (allr.fold == fold) & (allr.model == "SENTRY")]
        for _, sr in sub.iterrows():
            pan = [] if pd.isna(sr.panel) or sr.panel == "" else [apis.index(a) for a in str(sr.panel).split("|")]
            Xd = S.direct_features(d, pan)
            oof = np.zeros(len(tr))
            for a, b in inner:
                oof[b] = lgb.LGBMClassifier(**KW).fit(Xd[tr[a]], y_risk[tr[a]]).predict_proba(Xd[tr[b]])[:, 1]
            pt = lgb.LGBMClassifier(**KW).fit(Xd[tr], y_risk[tr]).predict_proba(Xd[te])[:, 1]
            mtr, mte = S.measured_part(d, tr, pan, pnec), S.measured_part(d, te, pan, pnec)
            lo = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
            free = ~mtr
            platt = LogisticRegression(C=1e3).fit(lo(oof[free])[:, None], y_risk[tr][free])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof[free], y_risk[tr][free])
            for name, q in [("Direct raw", pt), ("Direct + Platt", platt.predict_proba(lo(pt)[:, None])[:, 1]),
                            ("Direct + isotonic", iso.predict(pt))]:
                q = np.clip(np.where(mte, 1.0, q), 1e-4, 1 - 1e-4)
                rows.append(dict(seed=seed, fold=fold, strategy=sr.strategy, k=sr.k, model=name,
                                 AUC=roc_auc_score(y_risk[te], q), Brier=brier_score_loss(y_risk[te], q),
                                 LogLoss=log_loss(y_risk[te], q, labels=[0, 1])))
            rows.append(dict(seed=seed, fold=fold, strategy=sr.strategy, k=sr.k, model="SENTRY", AUC=sr.AUC,
                             Brier=sr.Brier, LogLoss=sr.LogLoss))
        print(f"seed {seed} fold {fold} done", flush=True)
res = pd.DataFrame(rows)
res.to_csv(R / "calibration_check.csv", index=False)
print(res.groupby("model")[["AUC", "Brier", "LogLoss"]].mean().round(4).to_string())
piv = res.pivot_table(index=["seed", "fold", "strategy", "k"], columns="model", values="LogLoss")
for m in ["Direct raw", "Direct + Platt", "Direct + isotonic"]:
    dd = piv["SENTRY"] - piv[m]
    print(f"SENTRY - {m}: mean dLogLoss {dd.mean():.4f}, SENTRY better in {(dd < 0).mean():.2f} of {len(dd)} panels")
