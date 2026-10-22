"""Robustness checks for the main rain effects (strict FE: plant x calendar month + state x
year-month, SEs clustered by state). One pollutant per call: robust_models.py <workdir> <group>

  lead        next month's heavy-rain days added (placebo: should be ~0)
  lag         last month's heavy-rain days added
  random      rain series of a randomly assigned plant in another state (placebo: ~0)
  p95/p999/50mm  alternative heavy-rain definitions
  balanced    plants reporting consistently every year
  reporting   missing reports (non-receipt) and other no-data codes as outcomes
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
W = Path(sys.argv[1])
G = sys.argv[2]
OUT = W / "upgrades"

cols = ["permit", "group", "ym", "cal_m", "year", "exceed", "logc", "n_heavy", "STATE_CODE"]
p = pd.concat([c[c["group"] == G] for c in pd.read_csv(W / "panel.csv.gz", usecols=cols, chunksize=1_000_000,
                                                       dtype={"permit": str, "ym": str, "STATE_CODE": str})])
clim = pd.read_csv(W / "climate_monthly.csv.gz", dtype={"permit": str, "ym": str}, usecols=["permit", "ym", "n_heavy"])
clim["ymp"] = pd.PeriodIndex(clim["ym"], freq="M")
lead = clim.assign(ym=(clim["ymp"] - 1).astype(str))[["permit", "ym", "n_heavy"]].rename(columns={"n_heavy": "nh_lead"})
lag = clim.assign(ym=(clim["ymp"] + 1).astype(str))[["permit", "ym", "n_heavy"]].rename(columns={"n_heavy": "nh_lag"})
# random-plant placebo: each plant gets the rain of a random plant in a different state
pl = p.drop_duplicates("permit")[["permit", "STATE_CODE"]].reset_index(drop=True)
allp = clim["permit"].unique()
st = pd.read_csv(W / "plants.csv", dtype={"permit": str}, usecols=["permit", "STATE_CODE"]).set_index("permit")["STATE_CODE"]
rng = np.random.default_rng(7)
donor = {}
for r in pl.itertuples():
    for _ in range(50):
        c = allp[rng.integers(len(allp))]
        if st.get(c) != r.STATE_CODE:
            donor[r.permit] = c
            break
rnd = clim[["permit", "ym", "n_heavy"]].rename(columns={"permit": "donor", "n_heavy": "nh_rand"})
p["donor"] = p["permit"].map(donor)
alt = pd.read_csv(OUT / "alt_rain.csv.gz", dtype={"permit": str, "ym": str})
rep = pd.read_csv(OUT / "reporting.csv.gz", dtype={"permit": str, "ym": str})
rep = rep[rep["group"] == G].drop(columns="group")
p = (p.merge(lead, on=["permit", "ym"], how="left").merge(lag, on=["permit", "ym"], how="left")
     .merge(rnd, on=["donor", "ym"], how="left").merge(alt, on=["permit", "ym"], how="left"))
p["pm"] = (p["permit"] + "_" + p["cal_m"].astype(str)).astype("category")
p["sym"] = (p["STATE_CODE"] + "_" + p["ym"]).astype("category")
b = lambda s, k: (s == k).astype(float) if k < 2 else (s >= 2).astype(float)
for v, nm in [("n_heavy", ""), ("nh_lead", "_lead"), ("nh_lag", "_lag"), ("nh_rand", "_rand")]:
    p[f"r1{nm}"], p[f"r2{nm}"] = b(p[v], 1), b(p[v], 2)
p["q1"], p["q2"], p["q3"] = (p["n95"] == 1) * 1.0, (p["n95"] == 2) * 1.0, (p["n95"] >= 3) * 1.0
p["x1"] = (p["n999"] >= 1) * 1.0
p["m1"] = (p["n50"] >= 1) * 1.0
# balanced: reported in >= 9 of 10 years, and no year below 75 % of the plant's typical count
# (complete calendar years 2016-2024 only; partial first/last years would exclude everyone)
yc = p[p["exceed"].notna() & p["year"].between(2016, 2024)].groupby(["permit", "year"]).size().unstack(fill_value=0)
bal = yc.index[((yc > 0).sum(1) == yc.shape[1]) & (yc.min(1) >= 0.75 * yc.median(1))]
p["balanced"] = p["permit"].isin(set(bal))

FE = "| pm + sym"
V = {"CRV1": "STATE_CODE"}
rows = []


def fit(name, y, fml, d):
    d = d[d[y].notna()]
    try:
        m = pf.feols(f"{y} ~ {fml} {FE}", data=d, vcov=V)
        t = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
        rows.append(t.assign(check=name, group=G, outcome=y, n=m._N, plants=d["permit"].nunique(), base=d[y].mean()))
    except Exception as e:
        print("FAILED", name, y, e, flush=True)


for y in ["exceed", "logc"]:
    fit("main", y, "r1 + r2", p)
    fit("lead placebo", y, "r1 + r2 + r1_lead + r2_lead", p)
    fit("lagged rain", y, "r1 + r2 + r1_lag + r2_lag", p)
    fit("random-plant placebo", y, "r1_rand + r2_rand", p)
    fit("95th percentile days", y, "q1 + q2 + q3", p)
    fit("99.9th percentile days", y, "x1", p)
    fit(">=50 mm days", y, "m1", p)
    fit("balanced plants", y, "r1 + r2", p[p["balanced"]])
# reporting gaps: own panel (months with no submitted value are not in the main panel)
rp = rep.merge(clim[["permit", "ym", "n_heavy"]], on=["permit", "ym"], how="inner")
rp = rp.merge(st.rename("STATE_CODE").reset_index(), on="permit", how="left")
rp["cal_m"] = rp["ym"].str[5:7]
rp["pm"] = (rp["permit"] + "_" + rp["cal_m"]).astype("category")
rp["sym"] = (rp["STATE_CODE"].astype(str) + "_" + rp["ym"]).astype("category")
rp["r1"], rp["r2"] = b(rp["n_heavy"], 1), b(rp["n_heavy"], 2)
for y in ["nonreceipt", "nodi_other"]:
    fit("reporting gaps", y, "r1 + r2", rp)
out = pd.concat(rows, ignore_index=True)
out.to_csv(OUT / f"robust_{G}.csv", index=False)
print(G, "balanced plants:", len(bal), "of", p["permit"].nunique(), "| rows", len(out), flush=True)
