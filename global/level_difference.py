"""Formal test: does the US rain response differ between advanced and secondary plants
(HydroWASTE-linked, outside combined-sewer cities)? Strict FE, SEs by state.
Output: out/level_difference.csv
"""
import os
import warnings
from pathlib import Path

import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
G = Path(os.environ.get("GLOBAL_DIR", "global"))
W = Path(os.environ.get("US_DIR", "wwtp_national"))
link = pd.read_csv(G / "out" / "link_us.csv", dtype={"permit": str})
link = link[link["LEVEL"].isin(["Secondary", "Advanced"])]
attr = pd.read_csv(W / "plants_attr.csv", dtype={"permit": str}, usecols=["permit", "cso_city"])
keep = set(link["permit"]) - set(attr.loc[attr["cso_city"] == 1, "permit"])
cols = ["permit", "group", "ym", "cal_m", "exceed", "n_heavy", "STATE_CODE"]
p = pd.concat([c[c["group"].isin(["ecoli", "fecal", "tss", "bod"]) & c["permit"].isin(keep) & c["exceed"].notna()]
               for c in pd.read_csv(W / "panel.csv.gz", usecols=cols, chunksize=1_000_000,
                                    dtype={"permit": str, "ym": str, "STATE_CODE": str})])
p = p.merge(link[["permit", "LEVEL"]], on="permit")
p["pm"] = (p["permit"] + "_" + p["cal_m"].astype(str)).astype("category")
p["sym"] = (p["STATE_CODE"] + "_" + p["ym"]).astype("category")
p["r1"], p["r2"] = (p["n_heavy"] == 1).astype(float), (p["n_heavy"] >= 2).astype(float)
p["adv"] = (p["LEVEL"] == "Advanced").astype(float)
p["r1_adv"], p["r2_adv"] = p["r1"] * p["adv"], p["r2"] * p["adv"]
rows = []
for g in ["ecoli", "fecal", "tss", "bod"]:
    d = p[p["group"] == g]
    m = pf.feols("exceed ~ r1 + r2 + r1_adv + r2_adv | pm + sym", data=d, vcov={"CRV1": "STATE_CODE"})
    t = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    rows.append(t.assign(group=g, plants=d["permit"].nunique()))
    print("done", g, flush=True)
r = pd.concat(rows)
r.to_csv(G / "out" / "level_difference.csv", index=False)
r["pp"], r["lo"], r["hi"] = 100 * r["Estimate"], 100 * r["2.5%"], 100 * r["97.5%"]
print(r[["group", "term", "pp", "lo", "hi", "Pr(>|t|)", "plants"]].round(3).to_string(index=False))
