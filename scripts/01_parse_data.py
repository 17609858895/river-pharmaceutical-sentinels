"""Parse the Wilkinson et al. (2022, PNAS) Global Pharmaceuticals workbook into tidy tables.

Outputs (data_processed/):
    compounds.csv      61 APIs with therapeutic class, LOD, LOQ, estimated value, lowest PNEC
    sites.csv          1,052 sampling sites with campaign/continent/site metadata
    conc_long.csv      site x API records with censoring status and [lower, upper] bounds (ng/L)
    conc_wide.csv      site x API matrix of reported values (ND -> NaN, e -> estimated value)
    country_socio.csv  country-level socioeconomic indicators from Table S9
    campaign_socio.csv campaign-level GNI / income class (Table S6) and regional population (Table S4)
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "Supplemental_Tables_PNAS_2022.xlsx"
OUT = ROOT / "data_processed"
OUT.mkdir(exist_ok=True)


def norm(s: str) -> str:
    s = str(s).lower().replace("&", "and")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------- S1 / S12
s1 = pd.read_excel(XLSX, sheet_name="Table S1", header=None)
rows, cls = [], None
for i in range(4, 65):
    c0, comp = s1.iat[i, 0], s1.iat[i, 1]
    if isinstance(c0, str) and c0.strip():
        cls = re.sub(r"\s+", " ", c0.replace("‎", " ")).strip()
    if isinstance(comp, str) and comp.strip():
        rows.append(dict(compound=comp.strip(), sub_class=cls,
                         LOD=to_float(s1.iat[i, 2]), LOQ=to_float(s1.iat[i, 3]),
                         est_value=to_float(s1.iat[i, 4])))
compounds = pd.DataFrame(rows)
antibiotic_subs = {"Lincosamide", "Macrolides", "Nitroimidazole", "Penicillin", "Sulfonamides", "Tetracyclines"}
compounds["therapeutic_class"] = [
    "Antibiotic" if (c.startswith("Antibiotics") or c in antibiotic_subs) else c for c in compounds["sub_class"]]

s12 = pd.read_excel(XLSX, sheet_name="Table S12", header=None)
pnec = {}
for i in range(5, 53):
    name = s12.iat[i, 0]
    if isinstance(name, str):
        pnec[name.strip()] = dict(PNEC_lowest=to_float(s12.iat[i, 6]),
                                  PNEC_AMR=np.nanmin([to_float(s12.iat[i, 4]), to_float(s12.iat[i, 5])])
                                  if not (np.isnan(to_float(s12.iat[i, 4])) and np.isnan(to_float(s12.iat[i, 5]))) else np.nan)
compounds["PNEC_lowest"] = compounds["compound"].map(lambda c: pnec.get(c, {}).get("PNEC_lowest", np.nan))
compounds["PNEC_AMR"] = compounds["compound"].map(lambda c: pnec.get(c, {}).get("PNEC_AMR", np.nan))

# --------------------------------------------------------------------------- S4 concentration matrix
s4 = pd.read_excel(XLSX, sheet_name="Table S4", header=None)
camp_cols = [j for j in range(2, s4.shape[1]) if isinstance(s4.iat[4, j], str) and s4.iat[4, j].strip()]
continent = s4.iloc[3, :].ffill()
camp_info = pd.DataFrame({
    "col": camp_cols,
    "campaign": [s4.iat[4, j].strip() for j in camp_cols],
    "continent": [continent.iat[j] for j in camp_cols],
    "n_sites": [int(s4.iat[5, j]) for j in camp_cols],
    "n_detected_reported": [to_float(s4.iat[6, j]) for j in camp_cols],
    "mean_cum_conc_reported": [to_float(s4.iat[7, j]) for j in camp_cols],
    "regional_population": [to_float(s4.iat[8, j]) for j in camp_cols],
})
camp_info["campaign_id"] = np.arange(len(camp_info))

comp_label = s4.iloc[:, 0].copy()
data_start = 10
end_row = s4.index[s4.iloc[:, 0].astype(str).str.startswith("Site-by-site")][0]
cur = None
records = []
lod_map = compounds.set_index("compound")[["LOD", "LOQ", "est_value"]].to_dict("index")
for i in range(data_start, end_row):
    lab = s4.iat[i, 0]
    if isinstance(lab, str) and lab.strip():
        cur = lab.strip()
    site = s4.iat[i, 1]
    if cur is None or pd.isna(site):
        continue
    site = int(site)
    for _, ci in camp_info.iterrows():
        if site > ci.n_sites:
            continue
        v = s4.iat[i, ci.col]
        lim = lod_map[cur]
        if isinstance(v, str):
            vs = v.strip()
            if vs == "ND":
                status, val, lo, hi = "ND", np.nan, 0.0, lim["LOD"]
            elif vs.lower().startswith("e"):
                status, val, lo, hi = "EST", to_float(vs[1:]), lim["LOD"], lim["LOQ"]
            else:
                status, val, lo, hi = "OTHER", np.nan, np.nan, np.nan
        elif pd.isna(v):
            status, val, lo, hi = "MISSING", np.nan, np.nan, np.nan
        else:
            status, val, lo, hi = "QUANT", float(v), float(v), float(v)
        records.append(dict(campaign_id=ci.campaign_id, site_num=site, compound=cur,
                            status=status, value=val, lower=lo, upper=hi))
conc = pd.DataFrame(records)
assert set(conc.compound) <= set(compounds.compound), set(conc.compound) - set(compounds.compound)

# reported site-by-site cumulative concentration (for checking)
cum_rows = []
for i in range(end_row + 1, end_row + 19):
    site = s4.iat[i, 1]
    if pd.isna(site):
        continue
    for _, ci in camp_info.iterrows():
        if int(site) <= ci.n_sites:
            cum_rows.append(dict(campaign_id=ci.campaign_id, site_num=int(site),
                                 cum_conc_reported=to_float(s4.iat[i, ci.col])))
cum = pd.DataFrame(cum_rows)

# --------------------------------------------------------------------------- S2 site metadata
s2 = pd.read_excel(XLSX, sheet_name="Table S2", header=None)
meta = s2.iloc[5:1057, :12].copy()
meta.columns = ["campaign_s2", "site_num", "date_raw", "time_raw", "river", "gps", "temp_C", "pH",
                "flow_m3s", "TDS_ppm", "EC_uScm", "notes"]
meta = meta[meta.campaign_s2.notna()].copy()
meta["campaign_s2"] = meta.campaign_s2.astype(str).str.strip()
# Some labels are irregular ("1 bis", "D"); S4 numbers sites 1..n in the same order as S2.
meta["site_label"] = meta.site_num.astype(str)
meta["site_num"] = meta.groupby("campaign_s2").cumcount() + 1
s2_counts = meta.groupby("campaign_s2").site_num.max()

# match S4 campaign names to S2 campaign names (name similarity + identical site count)
manual = {
    "King George Island": "Antarctica_ Antarctic Research Institute",
    "Congo_ Republic of_ Brazzaville": "Congo_ Brazzaville",
    "Cook Islands_ Rarotonga": "Cook Islands_ Raratonga",
    "Mexico_ Puerto Vallarta": "Mexio_ Puerto Vallarta",
}
s2_names = list(s2_counts.index)


def canon(s: str) -> str:
    s = norm(s)
    s = re.sub(r"^uk\b", "united kingdom", s)
    s = s.replace("drc", "dr").replace("republic of", "")
    return " ".join(s.split())


def name_score(a: str, b: str) -> float:
    a, b = canon(a), canon(b)
    ta, tb = set(a.split()), set(b.split())
    return difflib.SequenceMatcher(None, a, b).ratio() + len(ta & tb) / max(len(ta | tb), 1)


from scipy.optimize import linear_sum_assignment

S = np.array([[name_score(c, n) + 0.3 * (s2_counts[n] == k) for n in s2_names]
              for c, k in zip(camp_info.campaign, camp_info.n_sites)])
for c, n in manual.items():
    if c in set(camp_info.campaign):
        S[camp_info.index[camp_info.campaign == c][0], s2_names.index(n)] = 10.0
r, cidx = linear_sum_assignment(-S)
match = {camp_info.campaign_id.iat[i]: s2_names[j] for i, j in zip(r, cidx)}
camp_info["campaign_s2"] = camp_info.campaign_id.map(match)
camp_info["n_sites_s2"] = camp_info.campaign_s2.map(s2_counts)
camp_info["meta_misaligned"] = camp_info.n_sites != camp_info.n_sites_s2
camp_info["match_score"] = [difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
                            for a, b in zip(camp_info.campaign, camp_info.campaign_s2)]
dups = camp_info.campaign_s2[camp_info.campaign_s2.duplicated()]
if len(dups):
    print("WARNING duplicated matches:", camp_info[camp_info.campaign_s2.isin(dups)][["campaign", "campaign_s2", "n_sites"]])

site_frame = pd.DataFrame([(ci.campaign_id, k) for _, ci in camp_info.iterrows() for k in range(1, ci.n_sites + 1)],
                          columns=["campaign_id", "site_num"])
site_frame = site_frame.merge(camp_info[["campaign_id", "campaign", "campaign_s2", "continent", "n_sites",
                                         "regional_population", "meta_misaligned"]], on="campaign_id")
sites = site_frame.merge(meta, on=["campaign_s2", "site_num"], how="left")
# Where S2 and S4 list different numbers of sites, site-level metadata cannot be aligned safely:
# keep only campaign-level information (campaign centroid GPS, first sampling date).
mis = sites.meta_misaligned
camp_gps = meta.groupby("campaign_s2").gps.first()
camp_date = meta.groupby("campaign_s2").date_raw.first()
for c in ["time_raw", "river", "temp_C", "pH", "flow_m3s", "TDS_ppm", "EC_uScm", "notes", "site_label"]:
    sites.loc[mis, c] = np.nan
sites.loc[mis, "gps"] = sites.loc[mis, "campaign_s2"].map(camp_gps)
sites.loc[mis, "date_raw"] = sites.loc[mis, "campaign_s2"].map(camp_date)
sites["country"] = sites.campaign_s2.str.split("_").str[0].str.strip()


def parse_gps(s):
    nums = re.findall(r"-?\d+\.?\d*", str(s))
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else (np.nan, np.nan)


sites[["lat", "lon"]] = sites.gps.apply(lambda s: pd.Series(parse_gps(s)))
# A few partner GPS strings lost their decimal point (e.g. "-34836322"): restore it using the
# power of ten that brings the value closest to the campaign's other (valid) sites.
for col, lim in [("lat", 90), ("lon", 180)]:
    bad = sites[col].abs() > lim
    ref = sites[~bad].groupby("campaign_id")[col].median()
    for i in sites.index[bad]:
        v, r = sites.at[i, col], ref.get(sites.at[i, "campaign_id"], np.nan)
        cands = [v / 10 ** k for k in range(1, 10) if abs(v / 10 ** k) <= lim]
        sites.at[i, col] = min(cands, key=lambda c: abs(c - r)) if cands and not np.isnan(r) else np.nan
    print(f"repaired {int(bad.sum())} {col} values")
# Partners did not report GPS for three campaigns: use approximate city coordinates, flagged.
approx_city = {"Palestine_ West Bank": (31.90, 35.20), "Tanzania_ Dar es Salaam": (-6.79, 39.21),
               "Brunei_ Brunei River": (4.89, 114.94)}
sites["gps_approx"] = sites.lat.isna()
for k, (la, lo) in approx_city.items():
    msk = sites.campaign_s2.eq(k) & sites.lat.isna()
    sites.loc[msk, ["lat", "lon"]] = [la, lo]
camp_ll = sites.groupby("campaign_id")[["lat", "lon"]].transform("mean")
sites["lat"] = sites.lat.fillna(camp_ll.lat); sites["lon"] = sites.lon.fillna(camp_ll.lon)


def parse_date(v):
    if isinstance(v, pd.Timestamp) or hasattr(v, "year"):
        return pd.Timestamp(v)
    s = str(v)
    m = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", s)
    if not m:
        return pd.NaT
    d, mo, y = map(int, m.groups())
    y = y + 2000 if y < 100 else y
    try:
        return pd.Timestamp(year=y, month=mo, day=d)
    except ValueError:
        try:
            return pd.Timestamp(year=y, month=d, day=mo)
        except ValueError:
            return pd.NaT


sites["date"] = sites.date_raw.apply(parse_date)
for c in ["temp_C", "pH", "flow_m3s", "TDS_ppm", "EC_uScm"]:
    sites[c] = sites[c].apply(to_float)
sites["site_id"] = sites.campaign_id.astype(str) + "_" + sites.site_num.astype(str)
sites = sites.merge(cum, on=["campaign_id", "site_num"], how="left")
conc["site_id"] = conc.campaign_id.astype(str) + "_" + conc.site_num.astype(str)

# --------------------------------------------------------------------------- S6 / S9 socioeconomic
s6 = pd.read_excel(XLSX, sheet_name="Table S6", header=None)
s6r = []
for i in range(3, 140):
    name = s6.iat[i, 0]
    if isinstance(name, str) and ("_" in name or "," in name):
        s6r.append(dict(campaign_s6=name.strip(), GNI=to_float(s6.iat[i, 3]),
                        income_class=str(s6.iat[i, 4]).strip(), growth_rate=to_float(s6.iat[i, 5])))
s6d = pd.DataFrame(s6r)
manual6 = {"USA_ North Liberty": "USA_Iowa City", "USA_ Florida Eastern Intercoastal": "USA_ Florida"}
s6_names = list(s6d.campaign_s6)
S6 = np.array([[name_score(c, n) for n in s6_names] for c in camp_info.campaign])
for c, n in manual6.items():
    S6[camp_info.index[camp_info.campaign == c][0], s6_names.index(n)] = 10.0
r6, c6 = linear_sum_assignment(-S6)
s6_match = {camp_info.campaign_id.iat[i]: (s6_names[j], S6[i, j]) for i, j in zip(r6, c6)}
camp_info["campaign_s6"] = camp_info.campaign_id.map(lambda k: s6_match.get(k, (np.nan, 0))[0])
camp_info["s6_score"] = camp_info.campaign_id.map(lambda k: s6_match.get(k, (np.nan, 0))[1])
camp_info = camp_info.merge(s6d, on="campaign_s6", how="left")

s9 = pd.read_excel(XLSX, sheet_name="Table S9", header=None)
hdr = [str(h).strip() if not pd.isna(h) else None for h in s9.iloc[4].tolist()]
s9r = []
for i in range(5, 107):
    cname = s9.iat[i, 1]
    if isinstance(cname, str) and cname.strip():
        rec = {"country_s9": cname.strip()}
        for j in range(12, len(hdr)):
            if hdr[j]:
                rec[hdr[j]] = to_float(s9.iat[i, j])
        s9r.append(rec)
socio = pd.DataFrame(s9r)
socio = socio.loc[:, ~socio.columns.duplicated()]

# --------------------------------------------------------------------------- wide matrix + save
conc_valid = conc[conc.status.isin(["ND", "EST", "QUANT"])].copy()
conc_valid["value_reported"] = np.where(conc_valid.status == "ND", 0.0, conc_valid.value)
wide = conc_valid.pivot_table(index="site_id", columns="compound", values="value_reported", aggfunc="first")
sites = sites[sites.site_id.isin(wide.index)].reset_index(drop=True)
# Table S4 site totals include LOD-LOQ estimated values; reproduce that convention.
sites["cum_conc_calc"] = sites.site_id.map(
    conc_valid[conc_valid.status.isin(["QUANT", "EST"])].groupby("site_id").value.sum()).fillna(0.0)
sites["n_detected"] = sites.site_id.map(conc_valid[conc_valid.status != "ND"].groupby("site_id").size()).fillna(0)

compounds.to_csv(OUT / "compounds.csv", index=False)
sites.drop(columns=["gps"]).to_csv(OUT / "sites.csv", index=False)
conc_valid.to_csv(OUT / "conc_long.csv", index=False)
wide.to_csv(OUT / "conc_wide.csv")
socio.to_csv(OUT / "country_socio.csv", index=False)
camp_info.to_csv(OUT / "campaign_socio.csv", index=False)

# --------------------------------------------------------------------------- diagnostics
print("compounds:", len(compounds), "campaigns:", len(camp_info), "sites:", len(sites))
print("status counts:\n", conc.status.value_counts())
print("low-score campaign matches:\n",
      camp_info.loc[camp_info.match_score < 0.75, ["campaign", "campaign_s2", "n_sites", "n_sites_s2", "match_score"]].to_string())
print("misaligned site counts:\n", camp_info.loc[camp_info.meta_misaligned, ["campaign", "campaign_s2", "n_sites", "n_sites_s2"]].to_string())
print("low-score S6 matches:\n", camp_info.loc[camp_info.s6_score < 0.75, ["campaign", "campaign_s6", "s6_score"]].to_string())
chk = sites[["cum_conc_reported", "cum_conc_calc"]].dropna()
print("cumulative check |diff|>1%:", int((abs(chk.cum_conc_reported - chk.cum_conc_calc) > 0.01 * chk.cum_conc_reported + 1).sum()),
      "of", len(chk))
print("missing lat:", sites.lat.isna().sum(), "missing date:", sites.date.isna().sum())
print(sites[["temp_C", "pH", "flow_m3s", "TDS_ppm", "EC_uScm"]].notna().mean().round(3))
print("detection rate overall:", round((conc_valid.status != "ND").mean(), 3))
