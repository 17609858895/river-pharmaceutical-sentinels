"""Extra heuristic panel baselines on the same outer folds as 10_sentinel_v3.py.

union_heur: alternate the training-fold exceedance ranking and detection ranking (risk driver, load marker,
            risk driver, ...) - a no-search heuristic for the joint risk/load objective.
Evaluated with Direct-LGBM (k = 1..6) and SENTRY (k in EVAL_K), plus SENTRY on exceed_rank and expert panels.
Usage: python -u scripts/12_extra_baselines.py SEED [FOLD]
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
from sklearn.metrics import brier_score_loss, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import sentry_lib as S

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
EVAL_K = [1, 2, 3, 4, 6]
OUT = S.ROOT / "results" / "sentinel_v3"
OUT.mkdir(parents=True, exist_ok=True)
d = S.load()
apis = d["apis"]; n, p = d["det"].shape
targets = [j for j in range(p) if d["det"][:, j].sum() >= 5]
pnec, pamr = d["pnec"], d["pnec_amr"]
y_risk, y_cum, y_rq = S.observed_endpoints(d, np.arange(n), pnec)
y_amr = S.observed_endpoints(d, np.arange(n), pamr)[0]
EXPERT = [apis.index(a) for a in ["Caffeine", "Carbamazepine", "Sulfamethoxazole", "Atenolol", "Trimethoprim",
                                  "Gabapentin"]]
KW = dict(S.LGB_KW, n_jobs=S.NTHREAD)


def sc(pr, cum, rq, te, pa):
    pr = np.clip(pr, 1e-4, 1 - 1e-4)
    return dict(AUC=roc_auc_score(y_risk[te], pr), Brier=brier_score_loss(y_risk[te], pr),
                LogLoss=log_loss(y_risk[te], pr, labels=[0, 1]), R2_logcum=r2_score(y_cum[te], cum),
                R2_logRQ=r2_score(y_rq[te], rq), AUC_AMR=roc_auc_score(y_amr[te], pa))


ug = np.unique(d["groups"]); perm = np.random.default_rng(SEED).permutation(len(ug))
gshuf = pd.Series(d["groups"]).map(dict(zip(ug, perm))).values
rows = []
for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(n), groups=gshuf)):
    if ONLY is not None and fold != ONLY:
        continue
    inner = list(GroupKFold(3).split(np.zeros(len(tr)), groups=gshuf[tr]))
    freq = d["det"][tr].mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        exc = np.nan_to_num((d["v_obs"][tr] / pnec[None, :]) > 1).mean(0)
    pool = sorted({j for j in range(p) if freq[j] >= 0.05} & set(targets))
    fr = [j for j in np.argsort(-freq) if j in pool]; er = [j for j in np.argsort(-exc) if j in pool]
    union = []
    for a, b in zip(er, fr):
        for c in (a, b):
            if c not in union:
                union.append(c)
    strategies = {"union_heur": union[:6], "exceed_rank": er[:6], "expert": EXPERT}
    for strat, full in strategies.items():
        for k in range(1, 7):
            pan = full[:k]
            if strat == "union_heur":
                Xd = S.direct_features(d, pan)
                pr = np.where(S.measured_part(d, te, pan, pnec), 1.0,
                              lgb.LGBMClassifier(**KW).fit(Xd[tr], y_risk[tr]).predict_proba(Xd[te])[:, 1])
                cum = lgb.LGBMRegressor(**KW).fit(Xd[tr], y_cum[tr]).predict(Xd[te])
                rq = lgb.LGBMRegressor(**KW).fit(Xd[tr], y_rq[tr]).predict(Xd[te])
                pa = lgb.LGBMClassifier(**KW).fit(Xd[tr], y_amr[tr]).predict_proba(Xd[te])[:, 1]
                r = dict(seed=SEED, fold=fold, model="Direct-LGBM", strategy=strat, k=k,
                         panel="|".join(apis[j] for j in pan)); r.update(sc(pr, cum, rq, te, pa)); rows.append(r)
            if k in EVAL_K:
                st = S.structured_run(d, tr, te, inner, targets, pan, pnec, "aft")
                pa = S.site_endpoints_dep(d, te, targets, pan, st["mu_te"], st["sig"], pamr, st["rho"], n_draw=20)[0]
                r = dict(seed=SEED, fold=fold, model="SENTRY", strategy=strat, k=k, panel="|".join(apis[j] for j in pan),
                         rho=st["rho"]); r.update(sc(*st["dep"], te, pa)); rows.append(r)
        print(f"seed {SEED} fold {fold} {strat} done", flush=True)
    tag = f"seed{SEED}" + (f"_fold{ONLY}" if ONLY is not None else "")
    pd.DataFrame(rows).to_csv(OUT / f"extra_{tag}.csv", index=False)
    print(f"fold {fold} done", flush=True)
