"""Sample counts quoted in the text, in one file: plants, records, heavy-rain shares, combined-sewer
inventory, plants outside the contiguous US and dilution coverage. Output: upgrades/sample_counts.csv
Usage: python sample_counts.py <US workdir>
"""
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

W = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("US_DIR", "wwtp_national"))
SIX = ["ecoli", "fecal", "tss", "bod", "ammonia", "tp"]
rows = []
add = lambda k, v, note="": rows.append(dict(statistic=k, value=v, note=note))

p = pd.read_csv(W / "panel.csv.gz", usecols=["permit", "group", "exceed", "n_heavy"], dtype={"permit": str})
add("plants in panel (any parameter or flow)", p["permit"].nunique())
s = p[p["group"].isin(SIX)]
add("plant-month records, six parameters", len(s))
add("plants, six parameters", s["permit"].nunique())
e = s[s["exceed"].notna()]
add("records with a violation outcome, six parameters", len(e))
add("share of those records in months with 1 heavy-rain day", (e["n_heavy"] == 1).mean())
add("share of those records in months with >=2 heavy-rain days", (e["n_heavy"] >= 2).mean())
for g in SIX:
    add(f"plants in violation models: {g}", e.loc[e["group"] == g, "permit"].nunique())

pl = pd.read_csv(W / "plants.csv", dtype={"permit": str}, usecols=["permit", "STATE_CODE"])
pl = pl[pl["permit"].isin(set(p["permit"]))]
add("panel plants outside the contiguous US (AK, HI, PR, VI)", int(pl["STATE_CODE"].isin(["AK", "HI", "PR", "VI"]).sum()))
at = pd.read_csv(W / "plants_attr.csv", dtype={"permit": str})
at = at[at["permit"].isin(set(p["permit"]))]
add("panel plants in combined-sewer cities", int((at["cso_city"] == 1).sum()))
df = at["dilution_factor"]
add("panel plants with a dilution estimate", int(df.notna().sum()))
add("share of those with dilution factor < 10", (df < 10).sum() / df.notna().sum())

with zipfile.ZipFile(W / "raw" / "ALL_CSO_downloads.zip") as z:
    c = pd.read_csv(z.open(next(n for n in z.namelist() if n.lower().endswith(".csv"))), dtype=str, encoding="latin-1")
c.columns = c.columns.str.upper()
ch = c["PF_CHARACTER"].fillna("").str.upper().str.strip() if "PF_CHARACTER" in c else pd.Series("", index=c.index)
act = c[ch.str.contains("CSO") | ch.eq("TCS")]  # same definition as plant_attributes.py
nc = c[~ch.str.contains("CLS")]
add("CSO inventory: outfalls not coded as closed", len(nc), "number quoted in the introduction")
add("CSO inventory: permits with outfalls not coded as closed", nc["NPDES_ID"].nunique(), "number quoted in the introduction")
add("CSO inventory: outfalls coded CSO or TCS", len(act), "definition used for the active-outfall interaction test")
add("CSO inventory: permits with outfalls coded CSO or TCS", act["NPDES_ID"].nunique())
out = pd.DataFrame(rows)
out.to_csv(W / "upgrades" / "sample_counts.csv", index=False)
print(out.to_string(index=False))
