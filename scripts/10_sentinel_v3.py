"""v3 nested leave-campaign-out evaluation: sentinel panel design + panel-specific predictors.

Outer GroupKFold(5) over campaigns; inner GroupKFold(3) within outer-train for
  (i) greedy panel selection (engine: direct site-level LightGBM, out-of-fold on inner folds),
  (ii) out-of-fold scales/copula loading of the structured censored predictor,
  (iii) an initial empirical screening threshold; script 17 uses a separate three-way split.
Predictors evaluated on each selected panel (all retrained for the panel):
  Direct-LGBM       site-level boosters on (x, censored panel observations)
  SENTRY            pooled interval-censored AFT booster + per-API censored scales + one-factor censored copula
  SENTRY-indep      same, independent aggregation (copula ablation)
  Substitution      same structure, ND -> LOD/2 squared loss (censoring ablation)
  Hybrid            Direct-LGBM + SENTRY out-of-fold endpoint features (stacking)
Usage: python -u scripts/10_sentinel_v3.py SEED [FOLD]
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import sentry_lib as S

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
KMAX = 6
LAMS = [0.25, 0.5, 0.75]
EVAL_K = [1, 2, 3, 4, 6]
ALPHA = 0.10
QUICK = bool(os.environ.get("SENTRY_QUICK"))
if QUICK:
    KMAX, LAMS, EVAL_K = 2, [0.5], [1, 2]
OUT = S.ROOT / "results" / ("sentinel_v3_smoke" if QUICK else "sentinel_v3")
OUT.mkdir(parents=True, exist_ok=True)

d = S.load()
apis = d["apis"]; n, p = d["det"].shape
targets = [j for j in range(p) if d["det"][:, j].sum() >= 5]
pnec, pamr = d["pnec"], d["pnec_amr"]
y_risk, y_cum, y_rq = S.observed_endpoints(d, np.arange(n), pnec)
y_amr = S.observed_endpoints(d, np.arange(n), pamr)[0]
EXPERT = [apis.index(a) for a in ["Caffeine", "Carbamazepine", "Sulfamethoxazole", "Atenolol", "Trimethoprim",
                                  "Gabapentin"]]
SEL_KW = dict(S.LGB_KW, n_estimators=200, learning_rate=0.06, n_jobs=S.NTHREAD)
EVAL_KW = dict(S.LGB_KW, n_jobs=S.NTHREAD)


def shuffled(groups, seed):
    ug = np.unique(groups); perm = np.random.default_rng(seed).permutation(len(ug))
    return pd.Series(groups).map(dict(zip(ug, perm))).values


def direct_fit_predict(Xd, a, b, kw, extra_tr=None, extra_te=None, endpoints=("risk", "cum")):
    Xa, Xb = Xd[a], Xd[b]
    if extra_tr is not None:
        Xa, Xb = np.hstack([Xa, extra_tr]), np.hstack([Xb, extra_te])
    out = {}
    if "risk" in endpoints:
        out["risk"] = lgb.LGBMClassifier(**kw).fit(Xa, y_risk[a]).predict_proba(Xb)[:, 1]
    if "cum" in endpoints:
        out["cum"] = lgb.LGBMRegressor(**kw).fit(Xa, y_cum[a]).predict(Xb)
    if "rq" in endpoints:
        out["rq"] = lgb.LGBMRegressor(**kw).fit(Xa, y_rq[a]).predict(Xb)
    if "amr" in endpoints:
        out["amr"] = lgb.LGBMClassifier(**kw).fit(Xa, y_amr[a]).predict_proba(Xb)[:, 1]
    return out


def conformal_threshold(scores_pos, alpha):
    """Site-score lower-tail quantile; campaign-held-out performance is evaluated empirically."""
    s = np.sort(scores_pos); k = int(np.floor(alpha * (len(s) + 1)))
    return -np.inf if k < 1 else s[k - 1]


def scores(pr, cum, rq, te, pr_amr=None, thr=None):
    pr = np.clip(pr, 1e-4, 1 - 1e-4)
    r = dict(AUC=roc_auc_score(y_risk[te], pr), Brier=brier_score_loss(y_risk[te], pr),
             LogLoss=log_loss(y_risk[te], pr, labels=[0, 1]), R2_logcum=r2_score(y_cum[te], cum),
             R2_logRQ=r2_score(y_rq[te], rq))
    if pr_amr is not None and 0 < y_amr[te].sum() < len(te):
        r["AUC_AMR"] = roc_auc_score(y_amr[te], pr_amr)
    if thr is not None:
        pos = y_risk[te] == 1
        r["conf_miss"] = float((pr[pos] < thr).mean()) if pos.any() else np.nan
        r["conf_cleared"] = float((pr < thr).mean())
    return r


rows, panels_log = [], {}
gshuf = shuffled(d["groups"], SEED)
rng = np.random.default_rng(SEED + 99)
t0 = time.time()
for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(n), groups=gshuf)):
    if ONLY is not None and fold != ONLY:
        continue
    print(f"=== seed {SEED} fold {fold} ({time.time()-t0:.0f}s)", flush=True)
    inner = list(GroupKFold(3).split(np.zeros(len(tr)), groups=gshuf[tr]))
    freq = d["det"][tr].mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        exc = np.nan_to_num((d["v_obs"][tr] / pnec[None, :]) > 1).mean(0)
    pool = sorted({j for j in range(p) if freq[j] >= 0.05} & set(targets))
    freq_rank = [j for j in np.argsort(-freq) if j in pool]
    exc_rank = [j for j in np.argsort(-exc) if j in pool]

    # ------------------------------------------------ selection engine: inner-OOF direct predictions
    cache = {}

    def inner_oof(panel):
        key = tuple(sorted(panel))
        if key not in cache:
            Xd = S.direct_features(d, list(key))
            pr = np.zeros(len(tr)); cu = np.zeros(len(tr))
            for a, b in inner:
                o = direct_fit_predict(Xd, tr[a], tr[b], SEL_KW)
                pr[b], cu[b] = o["risk"], o["cum"]
            pr = np.where(S.measured_part(d, tr, list(key), pnec), 1.0, pr)
            cache[key] = (np.clip(pr, 1e-4, 1 - 1e-4), cu)
        return cache[key]

    def losses(panel):
        pr, cu = inner_oof(panel)
        return log_loss(y_risk[tr], pr, labels=[0, 1]), float(np.mean((y_cum[tr] - cu) ** 2))

    r0, c0 = losses([])

    def greedy(obj, lam=0.5):
        panel = []
        for _ in range(KMAX):
            best, bv = None, np.inf
            for c in pool:
                if c in panel:
                    continue
                lr, lc = losses(panel + [c])
                v = lr if obj == "risk" else lc if obj == "load" else lam * lr / r0 + (1 - lam) * lc / c0
                if v < bv:
                    best, bv = c, v
            panel.append(best)
        print(f"  greedy[{obj}{lam if obj == 'joint' else ''}]: {[apis[j] for j in panel]} ({time.time()-t0:.0f}s)",
              flush=True)
        return panel

    strategies = {"risk_greedy": greedy("risk"), "load_greedy": greedy("load")}
    for lam in LAMS:
        strategies[f"joint{lam}"] = greedy("joint", lam)
    strategies.update({"frequency": freq_rank[:KMAX], "exceed_rank": exc_rank[:KMAX], "expert": EXPERT[:KMAX]})
    for ri in range(5):
        strategies[f"random{ri}"] = list(rng.choice(pool, KMAX, replace=False))
    panels_log[fold] = {k: [apis[j] for j in v] for k, v in strategies.items()}

    def record(model, strategy, panel, res):
        r = dict(seed=SEED, fold=fold, model=model, strategy=strategy, k=len(panel),
                 panel="|".join(apis[j] for j in panel)); r.update(res); rows.append(r)

    # ------------------------------------------------ Direct-LGBM on every strategy and k
    for strat, full in [("none", [])] + list(strategies.items()):
        for k in ([0] if strat == "none" else range(1, KMAX + 1)):
            pan = full[:k]
            Xd = S.direct_features(d, pan)
            o = direct_fit_predict(Xd, tr, te, EVAL_KW, endpoints=("risk", "cum", "rq", "amr"))
            meas = S.measured_part(d, te, pan, pnec)
            pr = np.where(meas, 1.0, o["risk"])
            rr = scores(pr, o["cum"], o["rq"], te, o["amr"])
            # split conformal (campaign-level): train on inner-train campaigns, calibrate on the held-out
            # inner campaigns, apply the same model to the test campaigns
            a0, b0 = inner[0]
            mc = lgb.LGBMClassifier(**EVAL_KW).fit(Xd[tr[a0]], y_risk[tr[a0]])
            cal = np.where(S.measured_part(d, tr[b0], pan, pnec), 1.0, mc.predict_proba(Xd[tr[b0]])[:, 1])
            thr = conformal_threshold(cal[y_risk[tr[b0]] == 1], ALPHA)
            st = np.where(meas, 1.0, mc.predict_proba(Xd[te])[:, 1])
            pos = y_risk[te] == 1
            rr.update(conf_miss=float((st[pos] < thr).mean()), conf_cleared=float((st < thr).mean()),
                      conf_cleared_safe=float((st[~pos] < thr).mean()))
            record("Direct-LGBM", strat, pan, rr)
    print(f"  direct done ({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------ structured / hybrid on key strategies
    for strat in ["none", "risk_greedy", "joint0.5", "load_greedy"]:
        for k in ([0] if strat == "none" else EVAL_K):
            pan = strategies[strat][:k] if strat != "none" else []
            aft = S.structured_run(d, tr, te, inner, targets, pan, pnec, "aft")
            sub = S.structured_run(d, tr, te, inner, targets, pan, pnec, "sub")
            for name, res in [("SENTRY", aft["dep"]), ("SENTRY-indep", aft["ind"]), ("Substitution", sub["dep"])]:
                src = aft if name != "Substitution" else sub
                pa_amr = S.site_endpoints_dep(d, te, targets, pan, src["mu_te"], src["sig"], pamr, src["rho"], n_draw=20)[0]
                rr = scores(res[0], res[1], res[2], te, pa_amr); rr["rho"] = src["rho"]
                record(name, strat, pan, rr)
            Xd = S.direct_features(d, pan)
            o = direct_fit_predict(Xd, tr, te, EVAL_KW, aft["f_oof"][tr], aft["f_te"], ("risk", "cum", "rq", "amr"))
            pr = np.where(S.measured_part(d, te, pan, pnec), 1.0, o["risk"])
            record("Hybrid", strat, pan, scores(pr, o["cum"], o["rq"], te, o["amr"]))
        print(f"  structured {strat} done ({time.time()-t0:.0f}s)", flush=True)
    tag = f"seed{SEED}" + (f"_fold{ONLY}" if ONLY is not None else "")
    pd.DataFrame(rows).to_csv(OUT / f"v3_{tag}.csv", index=False)
    json.dump(panels_log, open(OUT / f"panels_{tag}.json", "w"), indent=1)
    print(f"  fold {fold} done ({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res["strategy"] = res.strategy.str.replace(r"random\d", "random", regex=True)
cols = [c for c in ["AUC", "Brier", "R2_logcum", "R2_logRQ", "AUC_AMR", "conf_miss", "conf_cleared", "conf_cleared_safe", "rho"] if c in res]
print(res.groupby(["model", "strategy", "k"])[cols].mean().round(3).to_string())
