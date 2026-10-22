"""Plant-level attributes for the upgrade analyses -> plants_attr.csv

* combined sewer overflow (CSO) system: outfalls active (CSO, TCS) and closed (CLS) in EPA's
  national CSO inventory, on the plant's own permit or in its city
* federal judicial enforcement on CSOs: case linked to the plant's facility, date entered
* receiving water (ATTAINS, latest cycle): designated drinking-water and recreation uses,
  recreation use not supported, pathogen impairment
* people within 3 miles (ACS via ECHO): population, low-income share, people of colour share
* dilution (dilution.py)
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
RAW = W / "raw"
rd = lambda z, n, **k: pd.read_csv(zipfile.ZipFile(RAW / z).open(n), dtype=str, encoding="latin-1", **k)

plants = pd.read_csv(W / "plants.csv", dtype={"permit": str})
keep = set((W / "dmr_permits.txt").read_text().split())
p = plants[plants["permit"].isin(keep)][["permit", "CITY", "STATE_CODE", "design_mgd", "actual_mgd",
                                         "major", "cso_permit", "cso_city"]].copy()
fac = rd("npdes_downloads.zip", "ICIS_FACILITIES.csv", usecols=["NPDES_ID", "FACILITY_UIN"])
p = p.merge(fac.drop_duplicates("NPDES_ID").rename(columns={"NPDES_ID": "permit",
                                                            "FACILITY_UIN": "registry_id"}), how="left")

# --- CSO inventory: active vs closed outfalls, by permit and by city
cso = rd("ALL_CSO_downloads.zip", "ALL_CSO_DOWNLOADS.csv")
ch = cso["PF_CHARACTER"].fillna("")
cso["closed"] = ch.eq("CLS").astype(int)
cso["active"] = (ch.str.contains("CSO") | ch.eq("TCS")).astype(int)
byp = cso.groupby("NPDES_ID")[["active", "closed"]].sum()
cso["city_key"] = cso["FACILITY_CITY"].fillna("").str.upper().str.strip() + "|" + cso["STATE_CODE"].fillna("")
byc = cso.groupby("city_key")[["active", "closed"]].sum()
p["city_key"] = p["CITY"].fillna("").str.upper().str.strip() + "|" + p["STATE_CODE"].fillna("")
p["cso_active_permit"] = p["permit"].map(byp["active"]).fillna(0)
p["cso_closed_permit"] = p["permit"].map(byp["closed"]).fillna(0)
p["cso_active_city"] = p["city_key"].map(byc["active"]).fillna(0)
p["cso_closed_city"] = p["city_key"].map(byc["closed"]).fillna(0)
tot = p["cso_active_city"] + p["cso_closed_city"]
p["cso_share_closed"] = (p["cso_closed_city"] / tot).where(tot > 0)

# --- federal judicial CSO cases and the date the decree was entered
ce = rd("case_downloads.zip", "CASE_ENFORCEMENTS.csv")
txt = (ce["CASE_NAME"].fillna("") + " " + ce["ENF_SUMMARY_TEXT"].fillna("")).str.lower()
cases = ce[txt.str.contains(r"combined sewer|\bcsos?\b|long[- ]term control plan")
           & ce["ACTIVITY_TYPE_DESC"].str.contains("Judicial", na=False)]["CASE_NUMBER"]
ms = rd("case_downloads.zip", "CASE_MILESTONES.csv")
ms = ms[ms["CASE_NUMBER"].isin(cases) & ms["SUB_ACTIVITY_TYPE_DESC"].str.contains("Entered", na=False)]
ms["date"] = pd.to_datetime(ms["ACTUAL_DATE"], errors="coerce")
entered = ms.groupby("CASE_NUMBER")["date"].min()
cf = rd("case_downloads.zip", "CASE_FACILITIES.csv")
cf = cf[cf["CASE_NUMBER"].isin(cases)].assign(entered=lambda x: x["CASE_NUMBER"].map(entered))
dec = cf.groupby("REGISTRY_ID")["entered"].min()
p["cso_decree"] = p["registry_id"].isin(dec.index).astype(int)
p["cso_decree_year"] = p["registry_id"].map(dec).dt.year

# --- receiving water (latest reporting cycle per permit)
au = rd("npdes_attains_downloads.zip", "NPDES_ATTAINS_AU_SUMMARIES.csv")
au = au[au["NPDES_ID"].isin(keep)].copy()
au["cyc"] = pd.to_numeric(au["REPORTINGCYCLE"], errors="coerce")
au = au[au["cyc"] == au.groupby("NPDES_ID")["cyc"].transform("max")]
g = au.groupby("NPDES_ID")
rw = pd.DataFrame({
    "rw_assessed": g.size().gt(0).astype(int),
    "rw_drinking_use": g["DRINKINGWATER_USE"].apply(lambda s: int(s.notna().any())),
    "rw_recreation_use": g["RECREATION_USE"].apply(lambda s: int(s.notna().any())),
    "rw_recreation_not_supporting": g["RECREATION_USE"].apply(lambda s: int(s.eq("Not Supporting").any())),
    "rw_pathogen_impaired": g["CAUSE_GROUPS_IMPAIRED"].apply(
        lambda s: int(s.fillna("").str.contains("PATHOGEN").any())),
})
p = p.merge(rw, left_on="permit", right_index=True, how="left")

# --- people within 3 miles (ACS 5-year via ECHO)
regs = set(p["registry_id"].dropna())
parts = []
with zipfile.ZipFile(RAW / "echo_demographics.zip") as z:
    for ch_ in pd.read_csv(z.open("ECHO_DEMOGRAPHICS.csv"), dtype=str, chunksize=500_000,
                           usecols=["REGISTRY_ID", "RADIUS_OF_AREA", "ACS_POPULATION", "ACS_IPOVBAS",
                                    "LOWINCOME", "MINORITY_POPULATION"]):
        parts.append(ch_[(ch_["RADIUS_OF_AREA"].str.strip() == "3") & ch_["REGISTRY_ID"].isin(regs)])
dm = pd.concat(parts).drop_duplicates("REGISTRY_ID")
for c in ["ACS_POPULATION", "ACS_IPOVBAS", "LOWINCOME", "MINORITY_POPULATION"]:
    dm[c] = pd.to_numeric(dm[c], errors="coerce")
dm["pop_3mi"] = dm["ACS_POPULATION"]
dm["lowincome_share"] = dm["LOWINCOME"] / dm["ACS_IPOVBAS"].where(dm["ACS_IPOVBAS"] > 0)
dm["poc_share"] = dm["MINORITY_POPULATION"] / dm["ACS_POPULATION"].where(dm["ACS_POPULATION"] > 0)
p = p.merge(dm[["REGISTRY_ID", "pop_3mi", "lowincome_share", "poc_share"]],
            left_on="registry_id", right_on="REGISTRY_ID", how="left").drop(columns="REGISTRY_ID")

# --- dilution
dil = pd.read_csv(W / "plants_dilution.csv", dtype={"permit": str})
p = p.merge(dil[["permit", "coastal", "dilution_factor", "dn_km_df100", "ww_ratio_at_outfall"]],
            on="permit", how="left")
p = p.drop(columns=["city_key"])
p.to_csv(W / "plants_attr.csv", index=False)

print(f"plants: {len(p):,}")
print(f"CSO city: {int(p['cso_city'].sum()):,}; with closed outfalls in city: {(p['cso_closed_city'] > 0).sum():,}; "
      f"federal CSO decree: {int(p['cso_decree'].sum()):,} (years {p['cso_decree_year'].min():.0f}-{p['cso_decree_year'].max():.0f})")
print(f"ATTAINS-linked: {p['rw_assessed'].notna().sum():,}; drinking-water use: {int(p['rw_drinking_use'].sum()):,}; "
      f"recreation use: {int(p['rw_recreation_use'].sum()):,}; pathogen-impaired: {int(p['rw_pathogen_impaired'].sum()):,}")
print(f"demographics: {p['pop_3mi'].notna().sum():,}; people within 3 mi: {p['pop_3mi'].sum() / 1e6:.1f} M")
