"""Projected change in rain-driven permit violations under local warming of +1, +2, +3 degC.

For each plant and pollutant: change in the share of months with 1 and >=2 heavy-rain days
(projection_shares.py, Clausius-Clapeyron scaling of the plant's own daily rain), weighted by
the calendar months the plant reports, times the strict-FE linear-probability effects (CSO-
specific). Coefficient uncertainty propagated with 2,000 normal draws.
Output: upgrades/projections.csv (national and by sewer type), upgrades/projection_plants.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
OUT = W / "upgrades"
G = ["ecoli", "fecal", "tss", "bod"]
rng = np.random.default_rng(1)
ND = 2000

res = pd.read_csv(OUT / "model_results.csv")
sh = pd.read_csv(OUT / "projection_shares.csv.gz", dtype={"permit": str})
base = sh[sh["dT"] == 0].set_index(["permit", "cal_m"])[["r1", "r2"]]
attr = pd.read_csv(W / "plants_attr.csv", dtype={"permit": str}, usecols=["permit", "cso_city", "STATE_CODE"])

rows, plant_rows = [], []
for g in G:
    r = res[(res["model"] == "cso") & (res["group"] == g) & (res["outcome"] == "exceed") & (res["how"] == "ols")]
    if r.empty:
        continue
    r = r.set_index("term")
    terms = ["r1", "r2", "r1:cso", "r2:cso"]
    mu = np.array([r.loc[t, "Estimate"] for t in terms])
    sd = np.array([r.loc[t, "Std. Error"] for t in terms])
    draws = rng.normal(mu, sd, size=(ND, 4))
    # months each plant reports this pollutant (per year), by calendar month
    pan = pd.read_csv(W / "panel.csv.gz", dtype={"permit": str}, usecols=["permit", "group", "cal_m", "year", "exceed"])
    pan = pan[(pan["group"] == g) & pan["exceed"].notna()]
    yrs = pan.groupby("permit")["year"].nunique()
    w = pan.groupby(["permit", "cal_m"]).size().div(yrs, level="permit").rename("w").reset_index()
    base_viol = pan.groupby("permit")["exceed"].sum().div(yrs)
    for dT in [1, 2, 3]:
        fut = sh[sh["dT"] == dT].set_index(["permit", "cal_m"])[["r1", "r2"]]
        d = (fut - base).rename(columns={"r1": "d1", "r2": "d2"}).reset_index()
        d = d.merge(w, on=["permit", "cal_m"])
        d["d1w"], d["d2w"] = d["d1"] * d["w"], d["d2"] * d["w"]
        pl = d.groupby("permit")[["d1w", "d2w"]].sum().reset_index().merge(attr, on="permit", how="left")
        pl["cso"] = pl["cso_city"].fillna(0)
        X1, X2, C = pl["d1w"].to_numpy(), pl["d2w"].to_numpy(), pl["cso"].to_numpy()
        # extra violation-months per year, per plant, for each coefficient draw
        extra = (np.outer(draws[:, 0], X1) + np.outer(draws[:, 1], X2)
                 + np.outer(draws[:, 2], X1 * C) + np.outer(draws[:, 3], X2 * C))
        bv = pl["permit"].map(base_viol).fillna(0).to_numpy()
        for name, m in {"all": np.ones(len(pl), bool), "combined-sewer city": C == 1,
                        "separate sewers": C == 0}.items():
            tot = extra[:, m].sum(1)
            rows.append(dict(group=g, dT=dT, plants=int(m.sum()), subset=name,
                             baseline_violation_months=bv[m].sum(),
                             extra_violation_months=np.median(tot), lo=np.quantile(tot, .025),
                             hi=np.quantile(tot, .975),
                             pct_change=100 * np.median(tot) / bv[m].sum(),
                             pct_lo=100 * np.quantile(tot, .025) / bv[m].sum(),
                             pct_hi=100 * np.quantile(tot, .975) / bv[m].sum()))
        pl["extra"] = extra.mean(0)
        pl["baseline"] = bv
        plant_rows.append(pl.assign(group=g, dT=dT)[["permit", "group", "dT", "cso", "extra", "baseline"]])

out = pd.DataFrame(rows)
out.to_csv(OUT / "projections.csv", index=False)
pd.concat(plant_rows).to_csv(OUT / "projection_plants.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(2).to_string(index=False))
