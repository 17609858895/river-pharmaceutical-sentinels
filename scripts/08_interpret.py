"""Interpretation of the fitted SENTRY booster (all sites; descriptive, not a performance estimate).

1 Sentinel information network: a pooled censored AFT booster trained with every candidate API observed
  except the target (leave-one-API-out conditioning; panel-specific training). Mean |TreeSHAP| of each
  observed API's censored observation (log value + detection flag) on the predicted ln-concentration of
  every other API. Rows = target, columns = source. A second booster with no panel gives covariate attribution.
2 Covariate attribution with no panel (k = 0): mean |SHAP| per covariate across targets, and
  signed SHAP vs feature value for the top covariates (binned SHAP dependence summaries).
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

import numpy as np
import pandas as pd
import xgboost as xgb

import sentry_lib as S

OUT = S.ROOT / "results" / "interpret"
OUT.mkdir(parents=True, exist_ok=True)
d = S.load()
apis = d["apis"]; n, p = d["det"].shape
targets = [j for j in range(p) if d["det"][:, j].sum() >= 5]
pool = [j for j in range(p) if d["det"][:, j].mean() >= 0.05]
sites = np.arange(n)
X, lb, ub, _ = S.build_long(d, sites, targets, [S.panel_mask(d, pool)])
b_full = S.fit_aft(X, lb, ub, rounds=300)
b_full.save_model(str(OUT / "sentry_allobserved.json"))
X, lb, ub, _ = S.build_long(d, sites, targets, [S.panel_mask(d, [])])
b_none = S.fit_aft(X, lb, ub, rounds=300)
b_none.save_model(str(OUT / "sentry_nopanel.json"))
ncov = d["X"].shape[1]
cov_names = list(d["X"].columns)


def shap_for(panel, b):
    O = S.obs_features(d, S.panel_mask(d, panel))
    A = S.api_descriptors(d, targets)
    res = []
    for t, j in enumerate(targets):
        Oj = O.copy(); Oj[:, j] = np.nan; Oj[:, j + p] = np.nan
        Xt = np.hstack([d["X"].values, Oj, np.repeat(A[t:t + 1], n, 0)])
        sv = b.predict(xgb.DMatrix(Xt, missing=np.nan), pred_contribs=True)[:, :-1]
        res.append((t, j, Xt, sv))
    return res


# 1. information network (all pool APIs observed)
net = pd.DataFrame(0.0, index=[apis[j] for j in targets], columns=[apis[j] for j in pool])
block = []
for t, j, Xt, sv in shap_for(pool, b_full):
    for a in pool:
        if a != j:
            net.loc[apis[j], apis[a]] = np.abs(sv[:, ncov + a]).mean() + np.abs(sv[:, ncov + p + a]).mean()
    block.append(dict(target=apis[j], covariates=np.abs(sv[:, :ncov]).sum(1).mean(),
                      panel=np.abs(sv[:, ncov:ncov + 2 * p]).sum(1).mean(),
                      api_identity=np.abs(sv[:, ncov + 2 * p:]).sum(1).mean()))
net.to_csv(OUT / "sentinel_information_network.csv")
pd.DataFrame(block).to_csv(OUT / "shap_block_share_allobserved.csv", index=False)

# 2. covariate attribution at k = 0
imp = np.zeros(ncov); dep = []
for t, j, Xt, sv in shap_for([], b_none):
    imp += np.abs(sv[:, :ncov]).mean(0)
    dep.append(pd.DataFrame({"target": apis[j], **{f"x_{c}": Xt[:, i] for i, c in enumerate(cov_names)},
                             **{f"s_{c}": sv[:, i] for i, c in enumerate(cov_names)}}))
imp = pd.Series(imp / len(targets), index=cov_names).sort_values(ascending=False)
imp.to_csv(OUT / "covariate_importance_k0.csv")
top = list(imp.index[:12])
dd = pd.concat(dep, ignore_index=True)
dd[["target"] + [f"x_{c}" for c in top] + [f"s_{c}" for c in top]].to_csv(OUT / "covariate_dependence_k0.csv", index=False)
print("top covariates (k=0):\n", imp.head(15).round(4).to_string())
print("strongest information links:")
st = net.stack().sort_values(ascending=False)
print(st.head(25).round(3).to_string())
