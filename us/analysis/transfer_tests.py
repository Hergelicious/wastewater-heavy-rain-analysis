"""Does the US rain response transfer? Heterogeneity of the heavy-rain effect on violations by
plant climate (terciles of mean annual precipitation and of mean temperature, 2014-2025), census
region and period (FY2016-2020 vs FY2021-2025); plus pollutant loads (log10 concentration x flow)
and flow. Main specification: plant x calendar month and state x year-month fixed effects, SEs by
state. Subgroup models give each group's effect; interaction models test the differences.
Usage: python transfer_tests.py <workdir> <group>   -> upgrades/transfer_tests_<group>.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
g = sys.argv[2]
REGION = {**{s: "Northeast" for s in "CT ME MA NH RI VT NJ NY PA".split()},
          **{s: "Midwest" for s in "IL IN MI OH WI IA KS MN MO NE ND SD".split()},
          **{s: "South" for s in "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX".split()},
          **{s: "West" for s in "AZ CO ID MT NV NM UT WY AK CA HI OR WA".split()}}

cols = ["permit", "group", "ym", "cal_m", "exceed", "logc", "flow_mgd", "n_heavy", "STATE_CODE"]
p = pd.concat([c[c["group"] == g] for c in pd.read_csv(W / "panel.csv.gz", usecols=cols, chunksize=1_000_000,
                                                         dtype={"permit": str, "ym": str, "STATE_CODE": str})])
clim = pd.read_csv(W / "climate_monthly.csv.gz", usecols=["permit", "ym", "tmean", "prcp_total"],
                   dtype={"permit": str, "ym": str})
pc = clim.groupby("permit").agg(prcp=("prcp_total", "mean"), tmean=("tmean", "mean"))
pc["prcp_annual_mm"] = pc["prcp"] * 12
pl = p[["permit"]].drop_duplicates().merge(pc, left_on="permit", right_index=True, how="left")
pl["wet_t"] = pd.qcut(pl["prcp_annual_mm"], 3, labels=["dry third", "middle third", "wet third"])
pl["warm_t"] = pd.qcut(pl["tmean"], 3, labels=["cold third", "middle third", "warm third"])
p = p.merge(pl[["permit", "wet_t", "warm_t", "prcp_annual_mm", "tmean"]], on="permit", how="left")
p["region"] = p["STATE_CODE"].map(REGION)
p["period"] = np.where(p["ym"] >= "2020-10", "FY2021-2025", "FY2016-2020")
p["pm"] = (p["permit"] + "_" + p["cal_m"].astype(str)).astype("category")
p["sym"] = (p["STATE_CODE"] + "_" + p["ym"]).astype("category")
p["r1"], p["r2"] = (p["n_heavy"] == 1).astype(float), (p["n_heavy"] >= 2).astype(float)
p["logflow"] = np.log10(p["flow_mgd"].where(p["flow_mgd"] > 0))
p["logload"] = p["logc"] + p["logflow"]
rows = []


def fit(d, y, fml, dim, level):
    d = d.dropna(subset=[y])
    try:
        m = pf.feols(f"{y} ~ {fml} | pm + sym", data=d, vcov={"CRV1": "STATE_CODE"})
    except Exception as exc:
        print("skip", dim, level, y, str(exc)[:80], flush=True)
        return
    t = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    rows.append(t.assign(group=g, dim=dim, level=level, outcome=y, n=m._N, plants=d["permit"].nunique(),
                         base=d[y].mean()))


# subgroup effects
for dim in ["wet_t", "warm_t", "region", "period"]:
    for level, d in p.groupby(dim, observed=True):
        if d["permit"].nunique() >= 100:
            fit(d, "exceed", "r1 + r2", dim, str(level))
    print("subgroups done", dim, flush=True)
# interaction tests (reference: middle tercile / Midwest / early period)
for dim, ref in [("wet_t", "middle third"), ("warm_t", "middle third"), ("region", "Midwest"),
                 ("period", "FY2016-2020")]:
    d = p.dropna(subset=[dim]).copy()
    terms = []
    for lv in sorted(set(d[dim].astype(str)) - {ref}):
        k = "".join(ch for ch in lv if ch.isalnum())
        ind = (d[dim].astype(str) == lv).astype(float)
        d[f"r1_x_{k}"], d[f"r2_x_{k}"] = d["r1"] * ind, d["r2"] * ind
        terms += [f"r1_x_{k}", f"r2_x_{k}"]
    fit(d, "exceed", "r1 + r2 + " + " + ".join(terms), dim, f"interaction vs {ref}")
    print("interaction done", dim, flush=True)
# loads and flow
fit(p, "logload", "r1 + r2", "load", "all")
fit(p, "logflow", "r1 + r2", "flow", "all")
fit(p[p["logload"].notna()], "logc", "r1 + r2", "concentration (load rows)", "all")
res = pd.concat(rows, ignore_index=True)
res.to_csv(W / "upgrades" / f"transfer_tests_{g}.csv", index=False)
print(res[["dim", "level", "outcome", "term", "Estimate", "2.5%", "97.5%", "Pr(>|t|)", "plants"]].round(4)
      .to_string(index=False), flush=True)
