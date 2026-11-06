"""Build site-level covariates, censored targets and risk endpoints.

Outputs (data_processed/):
    site_features.csv   covariates available without measuring any pharmaceutical
    site_targets.csv    log10 cumulative concentration, mixture risk endpoints, detection counts
    api_bounds_*.csv    site x API censoring bounds (lower / upper, ng/L) and status codes
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data_processed"

sites = pd.read_csv(D / "sites.csv", parse_dates=["date"])
conc = pd.read_csv(D / "conc_long.csv")
cmp_ = pd.read_csv(D / "compounds.csv")
socio = pd.read_csv(D / "country_socio.csv")
camp = pd.read_csv(D / "campaign_socio.csv")

# ----------------------------------------------------------------------------- country socio mapping
country_map = {"Gambia": "The Gambia", "Mexio": "Mexico", "United Kingdom": "UK", "Scotland": "UK"}
sites["country_s9"] = sites.country.replace(country_map)
sites.loc[sites.campaign_s2.str.startswith("Congo_ DR") | sites.campaign_s2.str.startswith("Congo_DR"),
          "country_s9"] = "Democratic Republic of the Congo"
sites.loc[sites.campaign_s2.eq("Congo_ Brazzaville"), "country_s9"] = "Republic of Congo"
sites.loc[sites.country.eq("Antarctica"), "country_s9"] = np.nan

short = {
    "Population": "pop_country", "Population below poverty line (%)": "poverty_pct",
    "Death rate (person/1000 population)": "death_rate", "Median Age": "median_age",
    "Unemployment rate": "unemployment", "Physician density (number per 1000 population)": "physician_density",
    "Dependency ratio": "dependency_ratio", "Age structure (% of above 65 years old)": "age65_pct",
    "Life expectancy (total)": "life_expectancy", "Real GDP (purchasing power parity, billion)": "gdp_ppp_bn",
    "Obesity (% of adult with BMI > 30.0)": "obesity_pct",
    "General government expenditure on health as a percentage of total government expenditure": "health_exp_pct",
    "Disability-adjusted life years (DALYs) attributable to the environment": "env_dalys",
    "School life expectancy (years of total population)": "school_life",
    "Hospital bed density (bed/1000 population)": "hospital_beds",
    "Prevalence of smoking any tobacco product among persons aged >= 15 years": "smoking_pct",
    "Produced municipal wastewater (km3/year or 109m3/year)": "ww_produced",
    "Treated municipal wastewater (km3/year or 109 m3/year)": "ww_treated",
    "Safely managed sanitation services (%)": "sanitation_pct",
    "Pharmacists (per 10,000 population)": "pharmacists",
    "Number of municipal wastewater treatment facilities": "n_wwtp",
    "Poverty headcount ratio at $5.50 a day (2011 PPP) (% of population)": "poverty550_pct",
    "School enrolment, tertiary (% gross)": "tertiary_enrol",
}
soc = socio.rename(columns=short)[["country_s9"] + [v for v in short.values()]].copy()
soc["gdp_pc"] = soc.gdp_ppp_bn * 1e9 / soc.pop_country
soc["ww_treated_frac"] = (soc.ww_treated / soc.ww_produced).clip(0, 1)
soc["ww_per_capita"] = soc.ww_produced * 1e9 / soc.pop_country
for c in ["pop_country", "gdp_ppp_bn", "gdp_pc", "env_dalys", "ww_produced", "ww_treated", "n_wwtp", "ww_per_capita"]:
    soc[c] = np.log10(soc[c].clip(lower=1e-9))
soc = soc.drop(columns=["ww_produced", "ww_treated"])
sites = sites.merge(soc, on="country_s9", how="left")

# ----------------------------------------------------------------------------- campaign-level
camp_cols = camp[["campaign_id", "GNI", "income_class", "growth_rate"]].copy()
camp_cols["income_ord"] = camp_cols.income_class.map({"Low": 0, "Lower-middle": 1, "Upper-middle": 2, "High": 3})
for lab, k in [("Low", 0), ("Lower", 1), ("Upper", 2), ("High", 3)]:
    camp_cols.loc[camp_cols.income_class.astype(str).str.startswith(lab), "income_ord"] = k
camp_cols["log_GNI"] = np.log10(camp_cols.GNI)
sites = sites.merge(camp_cols[["campaign_id", "log_GNI", "income_ord", "growth_rate"]], on="campaign_id", how="left")
sites["log_regional_pop"] = np.log10(pd.to_numeric(sites.regional_population, errors="coerce").where(lambda v: v > 0))

# ----------------------------------------------------------------------------- geography / time / design
sites["abs_lat"] = sites.lat.abs()
rad = np.deg2rad
sites["geo_x"] = np.cos(rad(sites.lat)) * np.cos(rad(sites.lon))
sites["geo_y"] = np.cos(rad(sites.lat)) * np.sin(rad(sites.lon))
sites["geo_z"] = np.sin(rad(sites.lat))
doy = sites.date.dt.dayofyear
# hemisphere-aligned season phase (day of year shifted by 182 d in the southern hemisphere)
doy_h = np.where(sites.lat < 0, (doy + 182) % 365, doy)
sites["season_sin"] = np.sin(2 * np.pi * doy_h / 365.25)
sites["season_cos"] = np.cos(2 * np.pi * doy_h / 365.25)
sites["year"] = sites.date.dt.year + sites.date.dt.dayofyear / 366
# Table S2 note: samples ordered upstream -> downstream within each campaign
sites["rel_position"] = (sites.site_num - 1) / (sites.n_sites - 1).replace(0, np.nan)
cent = sites.groupby("campaign_id")[["lat", "lon"]].transform("median")


def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.deg2rad, [la1, lo1, la2, lo2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


sites["log_dist_centroid_km"] = np.log10(haversine(sites.lat, sites.lon, cent.lat, cent.lon) + 0.1)
# implausible liquid-water temperature (likely air temperature) -> missing
sites.loc[sites.temp_C < -1, "temp_C"] = np.nan
for c in ["flow_m3s", "TDS_ppm", "EC_uScm"]:
    sites["log_" + c] = np.log10(sites[c].clip(lower=0) + 1)
for c in ["temp_C", "pH", "log_flow_m3s", "log_TDS_ppm", "log_EC_uScm"]:
    sites["miss_" + c] = sites[c].isna().astype(int)

# ----------------------------------------------------------------------------- field-note text indicators
txt = (sites.notes.fillna("") + " " + sites.river.fillna("")).str.lower()
lexicon = {
    "kw_wastewater": r"sewage|sewer|wwtp|stp\b|treatment plant|treatment works|effluent|outfall|discharg|drain|wastewater|waste water",
    "kw_upstream": r"upstream|before|pristine|source|spring|headwater|clean|natural|reserve|park|forest|mountain",
    "kw_downstream": r"downstream|after|below|confluence|mouth|estuar",
    "kw_urban": r"urban|city|town|residential|houses|market|centre|center|bridge|road",
    "kw_industrial": r"industr|factory|manufactur|pharma|plant",
    "kw_hospital": r"hospital|clinic",
    "kw_agri": r"agricult|farm|livestock|cattle|irrigat|crop|rice|pig|poultry|aquacult|fish farm",
    "kw_waste": r"waste|rubbish|garbage|dump|landfill|litter|plastic|trash",
    "kw_odour": r"smell|odou?r|foam|dirty|polluted|black|turbid|murky",
    "kw_lake_coast": r"lake|lagoon|reservoir|dam|sea|coast|beach|bay|harbou?r|canal",
    "kw_rural": r"rural|village|remote|countryside",
}
for k, pat in lexicon.items():
    sites[k] = txt.str.contains(pat, regex=True).astype(int)
sites["has_notes"] = sites.notes.notna().astype(int)

# ----------------------------------------------------------------------------- targets
conc = conc.merge(cmp_[["compound", "PNEC_lowest"]], on="compound", how="left")
conc["value_est"] = np.where(conc.status == "ND", 0.0, conc.value)
cum = conc.groupby("site_id").value_est.sum()
conc["RQ"] = conc.value_est / conc.PNEC_lowest
rq = conc.groupby("site_id").RQ.agg(RQ_max="max", RQ_sum="sum")
ndet = conc.assign(det=conc.status != "ND").groupby("site_id").det.sum()
tg = pd.DataFrame({"site_id": sites.site_id})
tg["log_cum"] = np.log10(tg.site_id.map(cum).fillna(0) + 1)
tg["n_detected"] = tg.site_id.map(ndet).fillna(0)
tg = tg.merge(rq, left_on="site_id", right_index=True, how="left")
tg["log_RQ_sum"] = np.log10(tg.RQ_sum + 1e-3)
tg["risk_any"] = (tg.RQ_max > 1).astype(int)

# ----------------------------------------------------------------------------- per-API censoring bounds
lo = conc.pivot_table(index="site_id", columns="compound", values="lower", aggfunc="first")
hi = conc.pivot_table(index="site_id", columns="compound", values="upper", aggfunc="first")
st = conc.pivot_table(index="site_id", columns="compound", values="status", aggfunc="first")
order = sites.site_id
lo, hi, st = lo.reindex(order), hi.reindex(order), st.reindex(order)

feat_cols = (["lat", "lon", "abs_lat", "geo_x", "geo_y", "geo_z", "season_sin", "season_cos", "year",
              "rel_position", "log_dist_centroid_km", "temp_C", "pH", "log_flow_m3s", "log_TDS_ppm", "log_EC_uScm"]
             + [c for c in sites.columns if c.startswith("miss_")]
             + ["log_regional_pop", "log_GNI", "income_ord", "growth_rate"]
             + [v for v in soc.columns if v != "country_s9"]
             + list(lexicon) + ["has_notes"])
meta_cols = ["site_id", "campaign_id", "campaign", "campaign_s2", "country", "country_s9", "continent",
             "site_num", "n_sites", "date", "gps_approx", "meta_misaligned"]
sites[meta_cols + feat_cols].to_csv(D / "site_features.csv", index=False)
tg.to_csv(D / "site_targets.csv", index=False)
lo.to_csv(D / "api_bounds_lower.csv"); hi.to_csv(D / "api_bounds_upper.csv"); st.to_csv(D / "api_status.csv")
print("features:", len(feat_cols), "sites:", len(sites))
print(tg.describe().round(3).to_string())
print(sites[list(lexicon)].mean().round(3).to_string())
