"""Panel-specific censored mixture predictors used by the current study.

A pooled lognormal AFT location model is fitted separately for each observed panel.
API scales and a censored one-factor Gaussian copula are estimated from inner
out-of-fold predictions. Measured components are plugged into mixture endpoints;
unmeasured eligible components are inferred. ND and EST records enter interval
likelihoods. The assay-wide observed endpoint and model eligibility differ; see README.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize_scalar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data_processed"
META = ["site_id", "campaign_id", "campaign", "campaign_s2", "country", "country_s9", "continent",
        "site_num", "n_sites", "date", "gps_approx", "meta_misaligned"]
NTHREAD = int(os.environ.get("SENTRY_NTHREAD", "12"))


# ----------------------------------------------------------------------------- data
def load():
    F = pd.read_csv(D / "site_features.csv")
    sid = F.site_id
    W = pd.read_csv(D / "conc_wide.csv", index_col=0).reindex(sid)
    ST = pd.read_csv(D / "api_status.csv", index_col=0).reindex(sid)
    cp = pd.read_csv(D / "compounds.csv").set_index("compound")
    X = F.drop(columns=META).replace([np.inf, -np.inf], np.nan)
    X = pd.concat([X, pd.get_dummies(F.continent, prefix="cont").astype(int)], axis=1)
    if "hw" in os.environ.get("SENTRY_EXTRA", ""):  # HydroWASTE wastewater-infrastructure covariates
        hw = pd.read_csv(D / "site_hydrowaste.csv").set_index("site_id").reindex(sid)
        X = pd.concat([X.reset_index(drop=True), hw.reset_index(drop=True)], axis=1)
    apis = list(W.columns)
    det = ((ST != "ND") & ST.notna()).values
    avail = ST.notna().values
    val = W.values.astype(float)
    lod = cp.loc[apis, "LOD"].values
    loq = cp.loc[apis, "LOQ"].values
    pnec = cp.loc[apis, "PNEC_lowest"].values
    pnec_amr = cp.loc[apis, "PNEC_AMR"].values
    status = ST.values
    lower = np.where(status == "ND", 0.0, np.where(status == "EST", lod[None, :], val))
    upper = np.where(status == "ND", lod[None, :], np.where(status == "EST", loq[None, :], val))
    # value used for observed endpoints (Table S4 convention: ND = 0, EST = estimated value)
    v_obs = np.where(det, val, 0.0)
    return dict(F=F, X=X, apis=apis, det=det, avail=avail, val=val, lod=lod, loq=loq, pnec=pnec,
                pnec_amr=pnec_amr, status=status, lower=lower, upper=upper, v_obs=v_obs,
                groups=F.campaign_id.values, cp=cp)


def obs_features(d, panel_mask):
    """Censored panel encoding. panel_mask: (n_sites, n_api) bool, True = measured.
    Returns (n_sites, 2*n_api): log10 value-or-LOD, detection flag; NaN where not measured."""
    lv = np.where(d["det"], np.log10(np.clip(d["val"], 1e-3, None)), np.log10(d["lod"])[None, :])
    dt = d["det"].astype(float)
    m = panel_mask & d["avail"]
    return np.hstack([np.where(m, lv, np.nan), np.where(m, dt, np.nan)])


def api_descriptors(d, targets):
    cp = d["cp"].loc[[d["apis"][j] for j in targets]]
    desc = pd.DataFrame({"logLOD": np.log10(cp.LOD.values)})
    cls = pd.get_dummies(cp.therapeutic_class.values, prefix="cls").astype(int)
    one = pd.get_dummies(pd.Categorical(targets), prefix="api").astype(int)
    return pd.concat([desc, cls, one], axis=1).values


# ----------------------------------------------------------------------------- long-format design
def build_long(d, sites, targets, masks):
    """masks: list of (n_sites_all, n_api) bool arrays (one per replicate)."""
    Xs = d["X"].values
    A = api_descriptors(d, targets)
    blocks, lb, ub, meta = [], [], [], []
    for r, M in enumerate(masks):
        O = obs_features(d, M)
        for t, j in enumerate(targets):
            s = sites[d["avail"][sites, j]]
            Oj = O[s].copy()
            Oj[:, j] = np.nan
            Oj[:, j + len(d["apis"])] = np.nan  # never see the target itself
            blocks.append(np.hstack([Xs[s], Oj, np.repeat(A[t:t + 1], len(s), 0)]))
            lb.append(d["lower"][s, j]); ub.append(d["upper"][s, j])
            meta.append(np.c_[s, np.full(len(s), j), np.full(len(s), r)])
    return np.vstack(blocks), np.concatenate(lb), np.concatenate(ub), np.vstack(meta)




def panel_mask(d, panel):
    M = np.zeros(d["det"].shape, bool)
    if len(panel):
        M[:, list(panel)] = True
    return M


# ----------------------------------------------------------------------------- learners
AFT_PARAMS = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal",
                  aft_loss_distribution_scale=2.0, tree_method="hist", learning_rate=0.06, max_depth=6,
                  min_child_weight=8, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0, max_bin=128)


def fit_aft(X, lb, ub, rounds=350, params=None):
    p = dict(AFT_PARAMS, **(params or {}), nthread=NTHREAD)
    dm = xgb.DMatrix(X, missing=np.nan)
    dm.set_float_info("label_lower_bound", lb)
    dm.set_float_info("label_upper_bound", ub)
    return xgb.train(p, dm, num_boost_round=rounds)


def fit_substitution(X, lb, ub, rounds=350, params=None):
    """Same booster, squared loss on ln C with ND -> LOD/2 and EST -> midpoint (common practice)."""
    y = np.where(lb == 0, np.log(ub / 2), np.where(lb < ub, np.log((lb + ub) / 2), np.log(np.clip(lb, 1e-6, None))))
    p = {k: v for k, v in AFT_PARAMS.items() if not k.startswith("aft") and k not in ("objective", "eval_metric")}
    p.update(objective="reg:squarederror", nthread=NTHREAD, **(params or {}))
    dm = xgb.DMatrix(X, label=y, missing=np.nan)
    return xgb.train(p, dm, num_boost_round=rounds)


def predict_mu(booster, d, sites, targets, panel, kind="aft"):
    """Return mu (ln scale) as (len(sites), len(targets)) for a fixed measured panel."""
    O = obs_features(d, panel_mask(d, panel))[sites]
    A = api_descriptors(d, targets)
    Xs = d["X"].values[sites]
    blocks = []
    for t, j in enumerate(targets):
        Oj = O.copy(); Oj[:, j] = np.nan; Oj[:, j + len(d["apis"])] = np.nan
        blocks.append(np.hstack([Xs, Oj, np.repeat(A[t:t + 1], len(sites), 0)]))
    pr = booster.inplace_predict(np.vstack(blocks), missing=np.nan)
    pr = pr.reshape(len(targets), len(sites)).T
    return np.log(np.clip(pr, 1e-12, None)) if kind == "aft" else pr


# ----------------------------------------------------------------------------- censored scale calibration
def censored_nll(mu, sigma, lb, ub):
    ll = np.log(np.clip(lb, 1e-12, None)); lu = np.log(ub)
    exact = lb == ub
    left = lb == 0
    inter = ~exact & ~left
    nll = np.zeros_like(mu)
    nll[exact] = -norm.logpdf(lu[exact], mu[exact], sigma) + lu[exact]
    nll[left] = -norm.logcdf((lu[left] - mu[left]) / sigma)
    nll[inter] = -np.log(np.clip(norm.cdf((lu[inter] - mu[inter]) / sigma) - norm.cdf((ll[inter] - mu[inter]) / sigma), 1e-300, None))
    return nll


def fit_sigma(mu, lb, ub):
    """1-D censored MLE of the scale with mu fixed (per-API heteroscedastic calibration)."""
    ok = ~np.isnan(lb)
    if ok.sum() < 10:
        return 1.0
    f = lambda ls: censored_nll(mu[ok], np.exp(ls), lb[ok], ub[ok]).mean()
    return float(np.exp(minimize_scalar(f, bounds=(np.log(0.2), np.log(6)), method="bounded").x))


# ----------------------------------------------------------------------------- endpoints
def p_exceed(mu, sigma, thr):
    """P(C > thr) under the lognormal predictive distribution (thr in ng/L)."""
    return 1 - norm.cdf((np.log(thr) - mu) / sigma)


def detected_mean(mu, sigma, lod):
    """E[C * 1(C > LOD)]: matches observed totals, where non-detects count as zero."""
    return np.exp(mu + sigma ** 2 / 2) * norm.cdf((mu + sigma ** 2 - np.log(lod)) / sigma)


def site_endpoints(d, sites, targets, panel, mu, sig, pnec, n_draw=200, seed=0):
    """Combine the measured panel (exact) with predictive draws for unmeasured APIs.
    Returns P(any RQ>1) (analytic), E[log10(cumulative+1)] and E[log10(sum RQ+1e-3)] (Monte Carlo),
    where drawn values below LOD count as zero, matching the observed-total convention."""
    panel = list(panel)
    vo = d["v_obs"][sites]
    ns = len(sites)
    meas_ex = np.zeros(ns, bool); meas_cum = np.zeros(ns); meas_rq = np.zeros(ns)
    for j in panel:
        a = d["avail"][sites, j]
        meas_cum += np.where(a, vo[:, j], 0)
        if not np.isnan(pnec[j]):
            meas_ex |= a & (vo[:, j] > pnec[j]); meas_rq += np.where(a, vo[:, j] / pnec[j], 0)
    rng = np.random.default_rng(seed)
    log_surv = np.zeros(ns)
    cum_d = np.repeat(meas_cum[:, None], n_draw, 1); rq_d = np.repeat(meas_rq[:, None], n_draw, 1)
    for t, j in enumerate(targets):
        if j in panel:
            continue
        c = np.exp(mu[:, t:t + 1] + sig[t] * rng.standard_normal((ns, n_draw)))
        c = np.where(c > d["lod"][j], c, 0.0) * d["avail"][sites, j][:, None]
        cum_d += c
        if not np.isnan(pnec[j]):
            # an observed exceedance requires detection: P(C > max(PNEC, LOD))
            pe = p_exceed(mu[:, t], sig[t], max(pnec[j], d["lod"][j])) * d["avail"][sites, j]
            log_surv += np.log1p(-np.clip(pe, 0, 1 - 1e-12))
            rq_d += c / pnec[j]
    p_any = np.where(meas_ex, 1.0, 1 - np.exp(log_surv))
    return p_any, np.log10(cum_d + 1).mean(1), np.log10(rq_d + 1e-3).mean(1)


def observed_endpoints(d, sites, pnec):
    vo = d["v_obs"][sites]
    with np.errstate(invalid="ignore", divide="ignore"):
        rq = np.where(np.isnan(pnec)[None, :], 0, vo / pnec[None, :])
    rq = np.nan_to_num(rq)
    return (rq.max(1) > 1).astype(int), np.log10(vo.sum(1) + 1), np.log10(rq.sum(1) + 1e-3)


# ----------------------------------------------------------------------------- dependence: censored one-factor copula
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(24)
GH_W = GH_W / GH_W.sum()


def _site_loglik_nodes(mu, sig, rho, lb, ub, avail):
    """log prod_j L_ij(eta_k) for each site i and quadrature node k. Shapes: mu (n,T), lb/ub (n,T)."""
    a = np.sqrt(rho) * sig[None, :, None]; b = np.sqrt(1 - rho) * sig[None, :, None]
    m = mu[:, :, None] + a * GH_X[None, None, :]
    ll = np.log(np.clip(lb, 1e-12, None))[:, :, None]; lu = np.log(np.clip(ub, 1e-12, None))[:, :, None]
    exact = (lb == ub)[:, :, None]; left = (lb == 0)[:, :, None]
    zu = (lu - m) / b; zl = (ll - m) / b
    out = np.where(exact, norm.logpdf(zu) - np.log(b),
                   np.where(left, norm.logcdf(zu),
                            np.log(np.clip(norm.cdf(zu) - norm.cdf(zl), 1e-300, None))))
    out = np.where(avail[:, :, None], out, 0.0)
    return out.sum(1)  # (n, K)


def fit_rho(mu, sig, lb, ub, avail):
    """Censored ML estimate of the shared site-factor loading rho in
    eps_ij = sqrt(rho) * eta_i + sqrt(1 - rho) * xi_ij (marginal scales sigma_j preserved)."""
    lb = np.nan_to_num(lb, nan=1.0); ub = np.nan_to_num(ub, nan=1.0)

    def nll(r):
        L = _site_loglik_nodes(mu, sig, r, lb, ub, avail)
        mx = L.max(1, keepdims=True)
        return -(mx[:, 0] + np.log((np.exp(L - mx) * GH_W[None, :]).sum(1))).sum()
    res = minimize_scalar(nll, bounds=(0.0, 0.95), method="bounded", options={"xatol": 1e-3})
    return float(res.x)


def site_endpoints_dep(d, sites, targets, panel, mu, sig, pnec, rho, n_draw=200, seed=0):
    """As site_endpoints, but unmeasured APIs share a latent site factor (one-factor Gaussian copula)."""
    panel = list(panel)
    vo = d["v_obs"][sites]; ns = len(sites)
    meas_ex = np.zeros(ns, bool); meas_cum = np.zeros(ns); meas_rq = np.zeros(ns)
    for j in panel:
        a = d["avail"][sites, j]
        meas_cum += np.where(a, vo[:, j], 0)
        if not np.isnan(pnec[j]):
            meas_ex |= a & (vo[:, j] > pnec[j]); meas_rq += np.where(a, vo[:, j] / pnec[j], 0)
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((ns, n_draw))
    sa, sb = np.sqrt(rho), np.sqrt(1 - rho)
    log_surv_nodes = np.zeros((ns, len(GH_X)))
    cum_d = np.repeat(meas_cum[:, None], n_draw, 1); rq_d = np.repeat(meas_rq[:, None], n_draw, 1)
    for t, j in enumerate(targets):
        if j in panel:
            continue
        av = d["avail"][sites, j]
        c = np.exp(mu[:, t:t + 1] + sig[t] * (sa * eta + sb * rng.standard_normal((ns, n_draw))))
        c = np.where(c > d["lod"][j], c, 0.0) * av[:, None]
        cum_d += c
        if not np.isnan(pnec[j]):
            thr = np.log(max(pnec[j], d["lod"][j]))
            pe = 1 - norm.cdf((thr - mu[:, t:t + 1] - sig[t] * sa * GH_X[None, :]) / (sig[t] * sb))
            log_surv_nodes += np.log1p(-np.clip(pe * av[:, None], 0, 1 - 1e-12))
            rq_d += c / pnec[j]
    p_any = 1 - (np.exp(log_surv_nodes) * GH_W[None, :]).sum(1)
    p_any = np.where(meas_ex, 1.0, p_any)
    return p_any, np.log10(cum_d + 1).mean(1), np.log10(rq_d + 1e-3).mean(1)


# ----------------------------------------------------------------------------- v3: panel-specific predictors
LGB_KW = dict(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=15, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, verbose=-1)


def direct_features(d, panel):
    return np.hstack([d["X"].values, obs_features(d, panel_mask(d, panel))])


def measured_part(d, sites, panel, pnec):
    ex = np.zeros(len(sites), bool)
    for j in panel:
        if not np.isnan(pnec[j]):
            ex |= d["avail"][sites, j] & (d["v_obs"][sites, j] > pnec[j])
    return ex


def fit_structured(d, sites, targets, panel, kind="aft", rounds=300):
    X, lb, ub, _ = build_long(d, sites, targets, [panel_mask(d, panel)])
    return (fit_aft if kind == "aft" else fit_substitution)(X, lb, ub, rounds=rounds)


def structured_params(d, mu, sites, targets, kind):
    sig = np.zeros(len(targets))
    for t, j in enumerate(targets):
        a = d["avail"][sites, j]; lb, ub = d["lower"][sites, j][a], d["upper"][sites, j][a]
        if kind == "aft":
            sig[t] = fit_sigma(mu[a, t], lb, ub)
        else:
            y = np.where(lb == 0, np.log(ub / 2), np.log(np.clip((lb + ub) / 2, 1e-6, None)))
            sig[t] = max(np.std(y - mu[a, t]), 0.2)
    T = targets
    rho = fit_rho(mu, sig, d["lower"][sites][:, T], d["upper"][sites][:, T], d["avail"][sites][:, T])
    return sig, rho


def structured_run(d, tr, te, inner, targets, panel, pnec, kind="aft", rounds=300):
    """Panel-specific structured censored predictor with out-of-fold scale/copula estimation.
    inner: list of (inner_train_idx, inner_val_idx) as positions in tr. Returns dict with test endpoints
    (dependent and independent aggregation) and OOF site features for training sites (for stacking)."""
    mus, vs = [], []
    for a, b in inner:
        bi = fit_structured(d, tr[a], targets, panel, kind, rounds)
        mus.append(predict_mu(bi, d, tr[b], targets, panel, kind)); vs.append(tr[b])
    mu_oof, s_oof = np.vstack(mus), np.concatenate(vs)
    sig, rho = structured_params(d, mu_oof, s_oof, targets, kind)
    bo = fit_structured(d, tr, targets, panel, kind, rounds)
    mu_te = predict_mu(bo, d, te, targets, panel, kind)
    dep = site_endpoints_dep(d, te, targets, panel, mu_te, sig, pnec, rho)
    ind = site_endpoints(d, te, targets, panel, mu_te, sig, pnec)

    def feats(mu, sites):
        pa, cum, rq = site_endpoints_dep(d, sites, targets, panel, mu, sig, pnec, rho, n_draw=100)
        pa = np.clip(pa, 1e-4, 1 - 1e-4)
        ndet = sum(p_exceed(mu[:, t], sig[t], d["lod"][j]) * d["avail"][sites, j]
                   for t, j in enumerate(targets) if j not in panel)
        return np.c_[np.log(pa / (1 - pa)), cum, rq, ndet]
    f_oof = np.full((len(d["F"]), 4), np.nan); f_oof[s_oof] = feats(mu_oof, s_oof)
    return dict(dep=dep, ind=ind, sig=sig, rho=rho, mu_te=mu_te, f_oof=f_oof, f_te=feats(mu_te, te))
