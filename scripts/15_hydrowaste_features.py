"""Site-level wastewater-infrastructure covariates from HydroWASTE v1.0 (Ehalt Macedo et al. 2022, ESSD).

For every sampling site, using operational / assumed-operational WWTP estimated outfalls:
  hw_log_dist_km        log10 distance to the nearest outfall (km)
  hw_n_{5,10,25,50}     number of outfalls within radius
  hw_logpop_{10,25,50}  log10(1 + population served by outfalls within radius)
  hw_logdis_{10,25}     log10(1 + treated discharge within radius, m3/d)
  hw_logload_df_{10,25} log10(1 + sum(population served / dilution factor)) within radius (dilution-adjusted load)
  hw_log_mindf_25       log10 of the smallest dilution factor within 25 km (NaN if none)
  hw_primary_share_25   share of served population with primary-only treatment within 25 km
  hw_advanced_share_25  share with advanced treatment within 25 km
  hw_nat_cov            national served population / national population (capped at 1.5)
  hw_source_national    1 if the country is covered by a national/regional WWTP dataset (SOURCE 1-11), else 0
Radius neighbourhoods ignore river topology (no flow direction), so they proxy for nearby urban wastewater pressure.
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
hw = pd.read_csv(ROOT / "external_data" / "hydrowaste" / "HydroWASTE_v10.csv", encoding="latin-1")
hw = hw[~hw.STATUS.isin(["Closed", "Decommissioned", "Non-Operational", "Projected", "Proposed", "Under Construction"])]
hw = hw.dropna(subset=["LAT_OUT", "LON_OUT"]).reset_index(drop=True)
sites = pd.read_csv(ROOT / "data_processed" / "site_features.csv")[["site_id", "country_s9", "lat", "lon", "pop_country"]]

R_EARTH = 6371.0
tree = BallTree(np.deg2rad(hw[["LAT_OUT", "LON_OUT"]].values), metric="haversine")
q = np.deg2rad(sites[["lat", "lon"]].values)
dist, _ = tree.query(q, k=1)
out = pd.DataFrame({"site_id": sites.site_id, "hw_log_dist_km": np.log10(dist[:, 0] * R_EARTH + 0.1)})
pop = hw.POP_SERVED.fillna(0).values
dis = hw.WASTE_DIS.fillna(0).values
df_ = hw.DF.values
lvl = hw.LEVEL.values
for r in [5, 10, 25, 50]:
    idx = tree.query_radius(q, r=r / R_EARTH)
    out[f"hw_n_{r}"] = [len(i) for i in idx]
    if r in (10, 25, 50):
        out[f"hw_logpop_{r}"] = [np.log10(1 + pop[i].sum()) for i in idx]
    if r in (10, 25):
        out[f"hw_logdis_{r}"] = [np.log10(1 + dis[i].sum()) for i in idx]
        out[f"hw_logload_df_{r}"] = [np.log10(1 + np.nansum(pop[i] / np.where(np.isfinite(df_[i]) & (df_[i] > 0), df_[i], np.nan)))
                                     for i in idx]
    if r == 25:
        out["hw_log_mindf_25"] = [np.log10(np.nanmin(df_[i])) if len(i) and np.isfinite(df_[i]).any() else np.nan for i in idx]
        tot = np.array([pop[i].sum() for i in idx])
        out["hw_primary_share_25"] = np.where(tot > 0, [pop[i][lvl[i] == "Primary"].sum() for i in idx] / np.maximum(tot, 1), np.nan)
        out["hw_advanced_share_25"] = np.where(tot > 0, [pop[i][lvl[i] == "Advanced"].sum() for i in idx] / np.maximum(tot, 1), np.nan)

# national coverage: match sites to HydroWASTE countries through the nearest outfall's country (robust to naming)
_, nn = tree.query(q, k=1)
site_cty = hw.COUNTRY.values[nn[:, 0]]
nat_pop = hw.groupby("COUNTRY").POP_SERVED.sum()
national_src = hw.groupby("COUNTRY").SOURCE.min()
pop_country = 10 ** sites.pop_country.values  # stored as log10 in site_features
out["hw_nat_cov"] = np.clip(pd.Series(site_cty).map(nat_pop).values / pop_country, 0, 1.5)
out["hw_source_national"] = (pd.Series(site_cty).map(national_src).values <= 11).astype(int)
far = dist[:, 0] * R_EARTH > 300  # nearest outfall in another region: national attributes not attributable
out.loc[far, ["hw_nat_cov"]] = np.nan
out.to_csv(ROOT / "data_processed" / "site_hydrowaste.csv", index=False)
print(f"WWTPs used: {len(hw):,}; sites: {len(out)}; nearest-outfall >300 km: {int(far.sum())}")
print(out.describe().T[["mean", "50%", "min", "max"]].round(2).to_string())
