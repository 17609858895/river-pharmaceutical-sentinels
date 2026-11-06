"""Site-level benchmark of 20 regressors/classifiers + stacking under leave-campaign-out CV.

Endpoints: log10 cumulative concentration (regression) and any-API PNEC exceedance (classification).
Inputs: site covariates, optionally plus a censored sentinel panel (--panel A,B,C).
Usage: python -u scripts/04_benchmark20.py --panel "" --tag cov
       python -u scripts/04_benchmark20.py --panel "Metformin,Sulfamethoxazole,Carbamazepine" --tag k3
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression,
                                  LogisticRegression, Ridge, RidgeCV, RidgeClassifier)
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

import sentry_lib as S

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--panel", default="")
ap.add_argument("--tag", default="cov")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--split", default="campaign", choices=["campaign", "random"],
                help="campaign = leave-campaign-out GroupKFold; random = row-level KFold (optimistic reference)")
args = ap.parse_args()
NJ = S.NTHREAD
OUT = S.ROOT / "results" / "benchmark20"
OUT.mkdir(parents=True, exist_ok=True)

d = S.load()
apis = d["apis"]
panel = [apis.index(a.strip()) for a in args.panel.split(",") if a.strip()]
X = np.hstack([d["X"].values, S.obs_features(d, S.panel_mask(d, panel))[:, [*panel, *[j + len(apis) for j in panel]]]]) \
    if panel else d["X"].values
names = list(d["X"].columns) + [f"{apis[j]}_log" for j in panel] + [f"{apis[j]}_det" for j in panel]
y_risk, y_cum, _ = S.observed_endpoints(d, np.arange(len(X)), d["pnec"])
groups = d["groups"]


def lin(m):
    # winsorise standardised inputs at +-5 SD: held-out campaigns can lie far outside the training range
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         FunctionTransformer(np.clip, kw_args={"a_min": -5, "a_max": 5}), m)


def tree(m):
    return make_pipeline(SimpleImputer(strategy="median"), m)


REG = {
    "OLS": lin(LinearRegression()),
    "Ridge": lin(Ridge(alpha=10)),
    "Lasso": lin(Lasso(alpha=0.02, max_iter=5000)),
    "ElasticNet": lin(ElasticNet(alpha=0.03, l1_ratio=0.5, max_iter=5000)),
    "BayesRidge": lin(BayesianRidge()),
    "Huber": lin(HuberRegressor(alpha=1e-3, max_iter=2000)),
    "PLS": lin(PLSRegression(n_components=8)),
    "KNN": lin(KNeighborsRegressor(n_neighbors=15, weights="distance")),
    "SVR": lin(SVR(C=3.0, epsilon=0.1, gamma="scale")),
    "GPR": lin(GaussianProcessRegressor(ConstantKernel() * RBF(10.0) + WhiteKernel(1.0), normalize_y=True)),
    "DecisionTree": tree(DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=0)),
    "RandomForest": tree(RandomForestRegressor(500, min_samples_leaf=3, max_features=0.4, n_jobs=NJ, random_state=0)),
    "ExtraTrees": tree(ExtraTreesRegressor(500, min_samples_leaf=3, max_features=0.5, n_jobs=NJ, random_state=0)),
    "GBDT": tree(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=0)),
    "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0),
    "AdaBoost": tree(AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=200, learning_rate=0.05, random_state=0)),
    "Bagging": tree(BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=3), n_estimators=200, n_jobs=NJ, random_state=0)),
    "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5, subsample=0.8,
                            colsample_bytree=0.7, reg_lambda=2, n_jobs=NJ, random_state=0),
    "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=NJ, random_state=0),
    "CatBoost": CatBoostRegressor(iterations=600, depth=5, learning_rate=0.05, l2_leaf_reg=5, verbose=0,
                                  thread_count=NJ, random_seed=0),
    "MLP": lin(MLPRegressor((64, 32), alpha=1e-2, learning_rate_init=1e-3, max_iter=800, early_stopping=True, random_state=0)),
}
CLF = {
    "Logistic": lin(LogisticRegression(C=0.5, max_iter=3000)),
    "Logistic-L1": lin(LogisticRegression(C=0.3, penalty="l1", solver="liblinear", max_iter=3000)),
    "Logistic-EN": lin(LogisticRegression(C=0.3, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=5000)),
    "LDA": lin(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "GaussianNB": lin(GaussianNB()),
    "KNN": lin(KNeighborsClassifier(n_neighbors=25, weights="distance")),
    "SVC": lin(SVC(C=1.0, probability=True, random_state=0)),
    "GPC": lin(GaussianProcessClassifier(ConstantKernel() * RBF(10.0), random_state=0)),
    "DecisionTree": tree(DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=0)),
    "RandomForest": tree(RandomForestClassifier(500, min_samples_leaf=3, max_features=0.4, n_jobs=NJ, random_state=0)),
    "ExtraTrees": tree(ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.5, n_jobs=NJ, random_state=0)),
    "GBDT": tree(GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=0)),
    "HistGB": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0),
    "AdaBoost": tree(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=200, learning_rate=0.05, random_state=0)),
    "Bagging": tree(BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=3), n_estimators=200, n_jobs=NJ, random_state=0)),
    "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5, subsample=0.8,
                             colsample_bytree=0.7, reg_lambda=2, n_jobs=NJ, random_state=0, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=NJ, random_state=0),
    "CatBoost": CatBoostClassifier(iterations=600, depth=5, learning_rate=0.05, l2_leaf_reg=5, verbose=0,
                                   thread_count=NJ, random_seed=0),
    "MLP": lin(MLPClassifier((64, 32), alpha=1e-2, learning_rate_init=1e-3, max_iter=800, early_stopping=True, random_state=0)),
    "RidgeClf": lin(RidgeClassifier(alpha=10)),
}
STACK_BASE = ["RandomForest", "ExtraTrees", "CatBoost", "LightGBM", "XGBoost", "SVR", "GPR", "Ridge"]
STACK_BASE_C = ["RandomForest", "ExtraTrees", "CatBoost", "LightGBM", "XGBoost", "SVC", "Logistic"]


def shuffled(groups, seed):
    ug = np.unique(groups); perm = np.random.default_rng(seed).permutation(len(ug))
    return pd.Series(groups).map(dict(zip(ug, perm))).values


def proba(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    s = m.decision_function(X)
    return 1 / (1 + np.exp(-s))


rows, preds = [], {}
t0 = time.time()
for rep in range(args.repeats):
    g = shuffled(groups, rep)
    if args.split == "random":
        outer = KFold(5, shuffle=True, random_state=rep).split(X)
    else:
        outer = GroupKFold(5).split(X, groups=g)
    for fold, (tr, te) in enumerate(outer):
        inner = list(GroupKFold(5).split(X[tr], groups=g[tr])) if args.split == "campaign" else             list(KFold(5, shuffle=True, random_state=rep).split(X[tr]))
        regs = dict(REG)
        regs["Stacking"] = StackingRegressor([(k, clone(REG[k])) for k in STACK_BASE],
                                             final_estimator=RidgeCV(alphas=np.logspace(-2, 3, 20)), cv=inner, n_jobs=1)
        for name, m in regs.items():
            mm = clone(m).fit(X[tr], y_cum[tr])
            p_te = np.ravel(mm.predict(X[te])); p_tr = np.ravel(mm.predict(X[tr]))
            rows.append(dict(task="log_cum", model=name, rep=rep, fold=fold,
                             R2=r2_score(y_cum[te], p_te), RMSE=mean_squared_error(y_cum[te], p_te) ** 0.5,
                             MAE=mean_absolute_error(y_cum[te], p_te), R2_train=r2_score(y_cum[tr], p_tr)))
            preds.setdefault(("log_cum", name, rep), np.zeros(len(X)))[te] = p_te
        clfs = dict(CLF)
        clfs["Stacking"] = StackingClassifier([(k, clone(CLF[k])) for k in STACK_BASE_C],
                                              final_estimator=LogisticRegression(C=1.0, max_iter=2000), cv=inner,
                                              stack_method="predict_proba", n_jobs=1)
        for name, m in clfs.items():
            mm = clone(m).fit(X[tr], y_risk[tr])
            p_te = np.clip(proba(mm, X[te]), 1e-4, 1 - 1e-4); p_tr = proba(mm, X[tr])
            rows.append(dict(task="risk_any", model=name, rep=rep, fold=fold, AUC=roc_auc_score(y_risk[te], p_te),
                             PRAUC=average_precision_score(y_risk[te], p_te), Brier=brier_score_loss(y_risk[te], p_te),
                             LogLoss=log_loss(y_risk[te], p_te), AUC_train=roc_auc_score(y_risk[tr], p_tr)))
            preds.setdefault(("risk_any", name, rep), np.zeros(len(X)))[te] = p_te
        print(f"rep {rep} fold {fold} done ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"bench20_{args.tag}_folds.csv", index=False)

res = pd.DataFrame(rows)
summ = res.groupby(["task", "model"]).agg(["mean", "std"])
summ.columns = ["_".join(c) for c in summ.columns]
summ.to_csv(OUT / f"bench20_{args.tag}_summary.csv")
pd.DataFrame({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in preds.items()}).assign(site_id=d["F"].site_id) \
    .to_csv(OUT / f"bench20_{args.tag}_oof.csv", index=False)
for t, col in [("log_cum", "R2_mean"), ("risk_any", "AUC_mean")]:
    print(summ.loc[t].sort_values(col, ascending=False).dropna(axis=1, how="all").round(3).to_string())
