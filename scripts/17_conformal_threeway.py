"""Split-conformal 'no exceedance' screening with a selection / calibration / test split of campaigns.

Within each outer training fold, campaigns are split (GroupKFold(3), first split) into
  selection campaigns  (2/3): panel selection (strategies below) and training of the screening model
  calibration campaigns (1/3): only used to set the threshold t = floor(alpha (n_pos + 1))-th smallest
                               score among at-risk calibration sites (empirical lower-tail screening)
and the threshold is applied to the outer test campaigns. Panels never see calibration or test campaigns.
Scores: Direct-LGBM risk probability, and SENTRY P(any RQ > 1); measured exceedances score 1.
Usage: python -u scripts/17_conformal_threeway.py SEED [FOLD]
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
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

import sentry_lib as S

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
ALPHAS = [0.05, 0.10, 0.20]
OUT = S.ROOT / "results" / "conformal"
OUT.mkdir(parents=True, exist_ok=True)
d = S.load()
apis = d["apis"]; n, p = d["det"].shape
targets = [j for j in range(p) if d["det"][:, j].sum() >= 5]
pnec = d["pnec"]
y_risk, y_cum, _ = S.observed_endpoints(d, np.arange(n), pnec)
SEL_KW = dict(S.LGB_KW, n_estimators=200, learning_rate=0.06, n_jobs=S.NTHREAD)
KW = dict(S.LGB_KW, n_jobs=S.NTHREAD)


def threshold(scores_pos, alpha):
    s = np.sort(scores_pos); k = int(np.floor(alpha * (len(s) + 1)))
    return -np.inf if k < 1 else s[k - 1]


def joint_greedy(sel, groups, pool, kmax=6, lam=0.5):
    inner = list(GroupKFold(3).split(np.zeros(len(sel)), groups=groups[sel]))
    cache = {}

    def losses(panel):
        key = tuple(sorted(panel))
        if key not in cache:
            Xd = S.direct_features(d, list(key)); pr = np.zeros(len(sel)); cu = np.zeros(len(sel))
            for a, b in inner:
                pr[b] = lgb.LGBMClassifier(**SEL_KW).fit(Xd[sel[a]], y_risk[sel[a]]).predict_proba(Xd[sel[b]])[:, 1]
                cu[b] = lgb.LGBMRegressor(**SEL_KW).fit(Xd[sel[a]], y_cum[sel[a]]).predict(Xd[sel[b]])
            pr = np.clip(np.where(S.measured_part(d, sel, list(key), pnec), 1.0, pr), 1e-4, 1 - 1e-4)
            cache[key] = (log_loss(y_risk[sel], pr, labels=[0, 1]), float(np.mean((y_cum[sel] - cu) ** 2)))
        return cache[key]
    r0, c0 = losses([])
    panel = []
    for _ in range(kmax):
        best = min((c for c in pool if c not in panel),
                   key=lambda c: lam * losses(panel + [c])[0] / r0 + (1 - lam) * losses(panel + [c])[1] / c0)
        panel.append(best)
    return panel


ug = np.unique(d["groups"]); perm = np.random.default_rng(SEED).permutation(len(ug))
gshuf = pd.Series(d["groups"]).map(dict(zip(ug, perm))).values
rows = []
for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(n), groups=gshuf)):
    if ONLY is not None and fold != ONLY:
        continue
    a0, b0 = next(GroupKFold(3).split(np.zeros(len(tr)), groups=gshuf[tr]))
    sel, cal = tr[a0], tr[b0]
    freq = d["det"][sel].mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        exc = np.nan_to_num((d["v_obs"][sel] / pnec[None, :]) > 1).mean(0)
    pool = sorted({j for j in range(p) if freq[j] >= 0.05} & set(targets))
    fr = [j for j in np.argsort(-freq) if j in pool]; er = [j for j in np.argsort(-exc) if j in pool]
    union = []
    for a, b in zip(er, fr):
        for c in (a, b):
            if c not in union:
                union.append(c)
    strategies = {"union_heur": union[:6], "exceed_rank": er[:6], "joint0.5": joint_greedy(sel, gshuf, pool)}
    inner_sel = list(GroupKFold(3).split(np.zeros(len(sel)), groups=gshuf[sel]))
    for strat, full in strategies.items():
        for k in [0, 3, 6]:
            if k == 0 and strat != "union_heur":
                continue
            pan = full[:k]
            Xd = S.direct_features(d, pan)
            clf = lgb.LGBMClassifier(**KW).fit(Xd[sel], y_risk[sel])
            scores = {"Direct-LGBM": {s: np.where(S.measured_part(d, idx, pan, pnec), 1.0, clf.predict_proba(Xd[idx])[:, 1])
                                      for s, idx in [("cal", cal), ("te", te)]}}
            both = np.concatenate([cal, te])
            st = S.structured_run(d, sel, both, inner_sel, targets, pan, pnec, "aft")
            ps = st["dep"][0]
            scores["SENTRY"] = {"cal": ps[:len(cal)], "te": ps[len(cal):]}
            for model, sc in scores.items():
                for alpha in ALPHAS:
                    t = threshold(sc["cal"][y_risk[cal] == 1], alpha)
                    pos = y_risk[te] == 1
                    rows.append(dict(seed=SEED, fold=fold, strategy="none" if k == 0 else strat, k=k, model=model,
                                     alpha=alpha, miss=float((sc["te"][pos] < t).mean()),
                                     cleared=float((sc["te"] < t).mean()), cleared_safe=float((sc["te"][~pos] < t).mean()),
                                     n_cal_pos=int((y_risk[cal] == 1).sum()), n_te_pos=int(pos.sum()),
                                     panel="|".join(apis[j] for j in pan)))
        print(f"seed {SEED} fold {fold} {strat} done", flush=True)
    tag = f"seed{SEED}" + (f"_fold{ONLY}" if ONLY is not None else "")
    pd.DataFrame(rows).to_csv(OUT / f"conformal_{tag}.csv", index=False)
