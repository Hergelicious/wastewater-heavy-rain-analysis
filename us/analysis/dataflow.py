"""Data flow of the US panel: plants and records after each selection step, by parameter.
Mirrors the panel step of wwtp_climate_pipeline.py (_panel_part / _series) with counts.
Usage: python dataflow.py <US workdir>  -> upgrades/dataflow.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import wwtp_climate_pipeline as w  # noqa: E402

W = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("US_DIR", "wwtp_national"))
SIX = ["ecoli", "fecal", "tss", "bod", "ammonia", "tp"]
cnt = {}


def add(step, g, rows=None, units=None, plants=None):
    k = (step, g)
    r = cnt.setdefault(k, dict(rows=0, units=0, plants=set()))
    r["rows"] += int(rows or 0)
    r["units"] += int(units or 0)
    if plants is not None:
        r["plants"] |= set(plants)


for f in sorted((W / "dmr").glob("dmr_*.csv.gz")):
    d = pd.read_csv(f, dtype=str)
    d = d.assign(group=d["PARAMETER_DESC"].map(w.classify))
    d = d[d["group"].isin(SIX)]
    for g, s in d.groupby("group"):
        add("1 effluent-gross DMR values for the parameter", g, len(s), None, s["EXTERNAL_PERMIT_NMBR"])
    d["ym"] = w.parse_dates(d["MONITORING_PERIOD_END_DATE"]).dt.to_period("M")
    d = d[d["ym"].notna()]
    d["val"], d["lim"] = w._num(d["DMR_VALUE_STANDARD_UNITS"]), w._num(d["LIMIT_VALUE_STANDARD_UNITS"])
    code = d["NODI_CODE"].fillna("").str.strip().str.upper()
    below = code == "B"
    nodi = (code != "") & ~below
    d["nd"] = below | d["DMR_VALUE_QUALIFIER_CODE"].fillna("").str.contains("<")
    d["enf"] = (d["lim"].notna() & (d["LIMIT_TYPE_CODE"].fillna("ENF").str.upper() == "ENF")
                & (d["OPTIONAL_MONITORING_FLAG"].fillna("N").str.upper() != "Y"))
    d = d.rename(columns={"EXTERNAL_PERMIT_NMBR": "permit"})
    ev = d[d["enf"] & ~nodi & (d["val"].notna() | below)]
    for g, s in ev.groupby("group"):
        add("2a violation outcome: values with an enforceable limit and a submitted result", g, len(s),
            s[["permit", "ym"]].drop_duplicates().shape[0], s["permit"])
    kind = d["group"].map({k: v[1] for k, v in w.GROUPS.items()})
    base = d[~nodi & (d["val"].notna() | below) & (d["VALUE_TYPE_CODE"].fillna("").str.upper().str[:1] == kind.fillna("?"))
             & (w._num(d["NMBR_OF_REPORT"]).fillna(1) == 1)]
    avg = base[base["STATISTICAL_BASE_TYPE_CODE"].fillna("").str.upper() == "AVG"].copy()
    for g, s in avg.groupby("group"):
        add("3a concentration: monthly-average values (single-month reports, no no-data code)", g, len(s),
            s[["permit", "ym"]].drop_duplicates().shape[0], s["permit"])
    key = ["PERM_FEATURE_NMBR", "PARAMETER_CODE", "STATISTICAL_BASE_CODE", "STANDARD_UNIT_DESC"]
    avg[key] = avg[key].fillna("")
    n = avg.groupby(["permit", "group"] + key).size().rename("n").reset_index()
    best = n.sort_values("n").groupby(["permit", "group"]).tail(1).drop(columns="n")
    one = avg.merge(best, on=["permit", "group"] + key)
    for g, s in one.groupby("group"):
        add("3b concentration: one dominant series per plant (outfall, code, statistic, unit)", g, len(s),
            s[["permit", "ym"]].drop_duplicates().shape[0], s["permit"])
    c, _ = w._series(avg.copy())
    for g, s in c.groupby("group"):
        add("3c concentration: series with at least 24 months", g, len(s), len(s), s["permit"])
    del d, avg, base, one, ev
    print("done", f.name, flush=True)

p = pd.read_csv(W / "panel.csv.gz", usecols=["permit", "group", "val", "exceed"], dtype={"permit": str})
p = p[p["group"].isin(SIX)]
for g, s in p.groupby("group"):
    e = s[s["exceed"].notna()]
    add("2b violation outcome: plant-months in the analysis panel (with weather)", g, len(e), len(e), e["permit"])
    v = s[s["val"].notna()]
    add("3d concentration: plant-months in the analysis panel (with weather)", g, len(v), len(v), v["permit"])
rows = []
for (step, g), r in sorted(cnt.items()):
    rows.append(dict(step=step, parameter=g, values=r["rows"], plant_months=r["units"] if r["units"] else np.nan,
                     plants=len(r["plants"])))
out = pd.DataFrame(rows)
# overall rows
pl = pd.read_csv(W / "plants.csv", dtype={"permit": str})
allp = pd.read_csv(W / "panel.csv.gz", usecols=["permit", "group"], dtype={"permit": str})
over = pd.DataFrame([
    dict(step="0a POTW permits (current version) with coordinates", parameter="all", plants=len(pl)),
    dict(step="0b ... with effluent-gross DMR values for any analysed parameter or flow, FY2016-2025", parameter="all",
         plants=len((W / "dmr_permits.txt").read_text().split())),
    dict(step="0c ... in the analysis panel (any parameter, flow or total N, with weather)", parameter="all",
         plants=allp["permit"].nunique()),
    dict(step="0d ... in the panel for at least one of the six analysed parameters", parameter="all",
         plants=allp[allp["group"].isin(SIX)]["permit"].nunique(), plant_months=int(allp["group"].isin(SIX).sum())),
])
out = pd.concat([over, out], ignore_index=True)
out.to_csv(W / "upgrades" / "dataflow.csv", index=False)
pd.set_option("display.width", 220)
print(out.to_string(index=False))
