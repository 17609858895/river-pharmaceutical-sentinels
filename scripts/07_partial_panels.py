"""Hidden-risk recovery for partial monitoring panels (v3: panel-specific predictors; leave-campaign-out 5 folds).

Real monitoring studies quantify different API subsets. For random and named panels we compare, for the
full 61-API mixture endpoint (any RQ > 1):
    Measured-only     risk flagged only if a measured API exceeds (current practice)
    Base-rate fill    measured part + unmeasured APIs at their training exceedance rates (no site information)
    Direct-LGBM       site-level classifier retrained for the panel
    SENTRY            panel-specific interval-censored AFT + censored one-factor copula
    Hybrid            Direct-LGBM + SENTRY out-of-fold features
Usage: python -u scripts/07_partial_panels.py [n_random_per_k] [fold]
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import sentry_lib as S

N_RAND = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
KS = [1, 2, 3, 5, 8, 12]
d = S.load()
apis = d["apis"]; n, p = d["det"].shape
targets = [j for j in range(p) if d["det"][:, j].sum() >= 5]
pnec = d["pnec"]
y_risk, y_cum, y_rq = S.observed_endpoints(d, np.arange(n), pnec)
OUT = S.ROOT / "results" / "partial_panels"
OUT.mkdir(parents=True, exist_ok=True)
NAMED = {
    "Wastewater markers": ["Caffeine", "Carbamazepine", "Sulfamethoxazole", "Atenolol", "Trimethoprim"],
    "Antibiotics (13)": ["Ciprofloxacin", "Enrofloxacin", "Lincomycin", "Clarithromycin", "Erythromycin", "Tylosin",
                         "Metronidazole", "Cloxacillin", "Sulfadiazine", "Sulfamethoxazole", "Trimethoprim",
                         "Oxytetracycline", "Tetracycline"],
    "CNS drugs": ["Carbamazepine", "Gabapentin", "Venlafaxine", "Desvenlafaxine", "Citalopram", "Amitriptyline",
                  "Tramadol", "Codeine"],
    "High-volume drugs": ["Metformin", "Paracetamol", "Caffeine", "Gabapentin", "Naproxen"],
}
KW = dict(S.LGB_KW, n_jobs=S.NTHREAD)

rows = []
ug = np.unique(d["groups"]); perm = np.random.default_rng(0).permutation(len(ug))
gshuf = pd.Series(d["groups"]).map(dict(zip(ug, perm))).values
rng = np.random.default_rng(123)
t0 = time.time()
for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(n), groups=gshuf)):
    panels = [(f"random-k{k}", list(rng.choice(targets, k, replace=False))) for k in KS for _ in range(N_RAND)]
    panels += [(nm, [apis.index(a) for a in lst]) for nm, lst in NAMED.items()]
    if ONLY is not None and fold != ONLY:
        continue
    inner = list(GroupKFold(3).split(np.zeros(len(tr)), groups=gshuf[tr]))
    with np.errstate(invalid="ignore", divide="ignore"):
        exc_rate = np.nan_to_num((d["v_obs"][tr] / pnec[None, :]) > 1).mean(0)
    for pname, panel in panels:
        ys = y_risk[te]
        meas = S.measured_part(d, te, panel, pnec)
        hidden = (ys == 1) & ~meas
        res = {"Measured-only": meas.astype(float)}
        surv = np.ones(len(te))
        for j in range(p):
            if j not in panel and not np.isnan(pnec[j]):
                surv *= 1 - exc_rate[j] * d["avail"][te, j]
        res["Base-rate fill"] = np.where(meas, 1.0, 1 - surv)
        st = S.structured_run(d, tr, te, inner, targets, panel, pnec, "aft")
        res["SENTRY"] = st["dep"][0]
        Xd = S.direct_features(d, panel)
        res["Direct-LGBM"] = np.where(meas, 1.0, lgb.LGBMClassifier(**KW).fit(Xd[tr], y_risk[tr]).predict_proba(Xd[te])[:, 1])
        Xh_tr = np.hstack([Xd[tr], st["f_oof"][tr]]); Xh_te = np.hstack([Xd[te], st["f_te"]])
        res["Hybrid"] = np.where(meas, 1.0, lgb.LGBMClassifier(**KW).fit(Xh_tr, y_risk[tr]).predict_proba(Xh_te)[:, 1])
        for m, pr in res.items():
            pr = np.clip(pr, 1e-4, 1 - 1e-4)
            thr = np.quantile(pr[ys == 0], 0.9) if (ys == 0).sum() else 0.5
            rows.append(dict(fold=fold, panel=pname, k=len(panel), method=m, AUC=roc_auc_score(ys, pr),
                             PRAUC=average_precision_score(ys, pr), Brier=brier_score_loss(ys, pr),
                             hidden_share=hidden.sum() / max(ys.sum(), 1),
                             hidden_recall_spec90=(pr[hidden] > thr).mean() if hidden.sum() else np.nan,
                             n_hidden=int(hidden.sum()), rho=st["rho"]))
    tag = "" if ONLY is None else f"_fold{ONLY}"
    pd.DataFrame(rows).to_csv(OUT / f"partial_panels{tag}.csv", index=False)
    print(f"fold {fold} done ({time.time()-t0:.0f}s)", flush=True)
res = pd.DataFrame(rows)
res["panel_type"] = np.where(res.panel.str.startswith("random"), "random", res.panel)
print(res.groupby(["panel_type", "k", "method"])[["AUC", "Brier", "hidden_share", "hidden_recall_spec90"]].mean()
      .round(3).to_string())
