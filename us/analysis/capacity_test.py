"""Hydraulic headroom: do plants that run close to their design flow respond more to heavy rain?
Utilisation = median ratio of monthly effluent flow to design flow over the plant's months without
a heavy-rain day (ratios outside (0, 3] treated as reporting errors; >= 12 dry months required).
Tests (main fixed effects: plant x calendar month, state x year-month; SEs by state):
  (1) effects by utilisation tercile; (2) interaction with 'near capacity' (utilisation >= 0.8);
  (3) interaction with utilisation tercile, adding an interaction with plant size (above-median design
      flow) so that headroom is not confused with size.
Usage: python capacity_test.py <workdir> <group>  -> upgrades/capacity_<group>.csv
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
cols = ["permit", "group", "ym", "cal_m", "exceed", "n_heavy", "STATE_CODE", "flow_mgd", "design_mgd"]
p = pd.concat([c[c["group"] == g] for c in pd.read_csv(W / "panel.csv.gz", usecols=cols, chunksize=1_000_000,
                                                         dtype={"permit": str, "ym": str, "STATE_CODE": str})])
ratio = p["flow_mgd"] / p["design_mgd"].where(p["design_mgd"] > 0)
dry = p.assign(ratio=ratio)[(p["n_heavy"] == 0) & (ratio > 0) & (ratio <= 3)]
u = dry.groupby("permit")["ratio"].agg(["median", "size"])
u = u[u["size"] >= 12]["median"].rename("util")
p = p.merge(u, left_on="permit", right_index=True, how="inner").dropna(subset=["exceed"])
pl = p.groupby("permit").agg(util=("util", "first"), design=("design_mgd", "first"))
pl["util_t"] = pd.qcut(pl["util"], 3, labels=["low", "middle", "high"])
pl["near_cap"] = (pl["util"] >= 0.8).astype(float)
pl["large"] = (pl["design"] >= pl["design"].median()).astype(float)
p = p.merge(pl[["util_t", "near_cap", "large"]], left_on="permit", right_index=True)
p["pm"] = (p["permit"] + "_" + p["cal_m"].astype(str)).astype("category")
p["sym"] = (p["STATE_CODE"] + "_" + p["ym"]).astype("category")
p["r1"], p["r2"] = (p["n_heavy"] == 1).astype(float), (p["n_heavy"] >= 2).astype(float)
for k in ["near_cap", "large"]:
    p[f"r1_{k}"], p[f"r2_{k}"] = p["r1"] * p[k], p["r2"] * p[k]
for lv in ["middle", "high"]:
    ind = (p["util_t"] == lv).astype(float)
    p[f"r1_{lv}"], p[f"r2_{lv}"] = p["r1"] * ind, p["r2"] * ind
rows = []


def fit(d, fml, label):
    m = pf.feols(f"exceed ~ {fml} | pm + sym", data=d, vcov={"CRV1": "STATE_CODE"})
    t = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    rows.append(t.assign(group=g, spec=label, n=m._N, plants=d["permit"].nunique(), base=d["exceed"].mean()))


for lv, d in p.groupby("util_t", observed=True):
    fit(d, "r1 + r2", f"utilisation tercile: {lv}")
fit(p, "r1 + r2 + r1_near_cap + r2_near_cap", "interaction: near capacity (>=0.8)")
fit(p, "r1 + r2 + r1_middle + r2_middle + r1_high + r2_high", "interaction: tercile vs low")
fit(p, "r1 + r2 + r1_high + r2_high + r1_middle + r2_middle + r1_large + r2_large",
    "interaction: tercile vs low, controlling size")
res = pd.concat(rows, ignore_index=True)
cut = pl.groupby("util_t", observed=True)["util"].agg(["min", "max", "size"])
res["tercile_bounds"] = "; ".join(f"{i}: {r['min']:.2f}-{r['max']:.2f} (n={r['size']})" for i, r in cut.iterrows())
res["share_near_cap"] = pl["near_cap"].mean()
res.to_csv(W / "upgrades" / f"capacity_{g}.csv", index=False)
pd.set_option("display.width", 220)
print(cut, "\nnear capacity share:", round(pl["near_cap"].mean(), 3))
print(res[["spec", "term", "Estimate", "2.5%", "97.5%", "Pr(>|t|)", "plants"]].assign(
    Estimate=lambda x: 100 * x["Estimate"], **{"2.5%": lambda x: 100 * x["2.5%"], "97.5%": lambda x: 100 * x["97.5%"]})
      .round(3).to_string(index=False), flush=True)
