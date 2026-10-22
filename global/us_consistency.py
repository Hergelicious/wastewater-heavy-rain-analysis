"""Consistency check at US plants: CMIP6 change in heavy-rain months (15 models, 2041-2060 vs
1995-2014) against scaling each plant's observed Daymet rain by 7% per degC (projection_shares).
Output: out/us_consistency.csv
"""
import os
from pathlib import Path

import pandas as pd

G = Path(os.environ.get("GLOBAL_DIR", "global"))
US = Path(os.environ.get("US_DIR", "wwtp_national"))
link = pd.read_csv(G / "out" / "link_us.csv", dtype={"permit": str})[["permit", "WASTE_ID"]]
cells = pd.read_csv(G / "out" / "hydrowaste_cells.csv")
sh = pd.read_csv(G / "out" / "cmip6_shares.csv")
m = link.merge(cells, on="WASTE_ID")
c = sh.merge(m[["permit", "cell"]], on="cell")
rows = []
for scen in ["ssp245", "ssp585"]:
    for col in ["share_1", "share_2plus"]:
        p = c.pivot_table(index=["permit", "model"], columns="scenario", values=col)
        by_model = p.groupby("model")[[scen, "historical"]].mean()
        r = 100 * (by_model[scen] / by_model["historical"] - 1)
        rows.append(dict(source="CMIP6 " + scen, measure=col, pct=r.mean(), lo=r.min(), hi=r.max(),
                         plants=p.index.get_level_values(0).nunique()))
ps = pd.read_csv(US / "upgrades" / "projection_shares.csv.gz", dtype={"permit": str})
ps = ps[ps["permit"].isin(set(m["permit"]))]
g = ps.groupby("dT")[["r1", "r2"]].mean()
for dT in [1, 2, 3]:
    for col, k in [("share_1", "r1"), ("share_2plus", "r2")]:
        rows.append(dict(source=f"7%/degC scaling, +{dT} degC", measure=col,
                         pct=100 * (g.loc[dT, k] / g.loc[0, k] - 1), plants=ps["permit"].nunique()))
out = pd.DataFrame(rows)
out.to_csv(G / "out" / "us_consistency.csv", index=False)
print(out.round(1).to_string(index=False))
