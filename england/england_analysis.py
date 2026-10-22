"""England replication test (criteria: england_preregistration.md).
Inputs: raw/ea_points.csv, raw/ea_obs/*.csv.gz, raw/rr_ens_mean_0.1deg_reg_2011-2025_v33.0e.nc
Outputs: out/england_results.csv, out/england_panel.csv.gz
"""
import os
import gzip
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import xarray as xr
from pyproj import Transformer

warnings.filterwarnings("ignore")
G = Path(os.environ.get("GLOBAL_DIR", "global"))
RAW, OUT = G / "raw", G / "out"
OUT.mkdir(exist_ok=True)
DET = {"0085": "bod", "0111": "ammonia", "0135": "tss"}

# ---- samples
pts = pd.read_csv(RAW / "ea_points.csv")
parts = []
for f in sorted((RAW / "ea_obs").glob("*.csv.gz")):
    try:
        d = pd.read_csv(f, dtype=str)
    except Exception:
        continue
    if len(d):
        parts.append(d)
obs = pd.concat(parts, ignore_index=True)
obs.columns = ["point", "time", "purpose", "det", "result", "unit"][: len(obs.columns)]
obs = obs[obs["det"].isin(DET)]
obs["group"] = obs["det"].map(DET)
r = obs["result"].astype(str).str.strip()
obs["nd"] = r.str.startswith("<")
obs["val"] = pd.to_numeric(r.str.lstrip("<>").str.strip(), errors="coerce")
obs.loc[obs["nd"], "val"] = obs.loc[obs["nd"], "val"] / 2
obs["date"] = pd.to_datetime(obs["time"].str[:10], errors="coerce")
obs = obs[obs["val"].notna() & (obs["val"] >= 0) & obs["date"].notna()]
obs = obs.groupby(["point", "group", "date"]).agg(val=("val", "mean"), nd=("nd", "max"),
                                                  purpose=("purpose", "first")).reset_index()
print(f"samples: {len(obs):,} from {obs['point'].nunique():,} works", flush=True)

# ---- rain at each works
tr = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
pts = pts[pts["point"].isin(obs["point"].unique())].dropna(subset=["easting", "northing"]).copy()
pts["lon"], pts["lat"] = tr.transform(pts["easting"].astype(float).values, pts["northing"].astype(float).values)
ds = xr.open_dataset(RAW / "rr_ens_mean_0.1deg_reg_2011-2025_v33.0e.nc")
sub = ds["rr"].sel(latitude=slice(49.5, 56.5), longitude=slice(-7, 2.5))
rr = sub.values  # time x lat x lon
lats, lons, times = sub.latitude.values, sub.longitude.values, pd.to_datetime(sub.time.values)
valid = ~np.isnan(rr).all(0)
vi, vj = np.where(valid)
cell = {}
for p in pts.itertuples():
    i, j = np.abs(lats - p.lat).argmin(), np.abs(lons - p.lon).argmin()
    if not valid[i, j]:  # coastal works: nearest land cell
        k = np.argmin((vi - i) ** 2 + (vj - j) ** 2)
        i, j = vi[k], vj[k]
    cell[p.point] = (i, j)
rain_rows = []
for p, (i, j) in cell.items():
    s = pd.Series(rr[:, i, j], index=times).fillna(0)
    p99 = s[s > 0.1].quantile(0.99)
    heavy = (s > p99).astype(int)
    rain_rows.append(pd.DataFrame({
        "point": p, "date": times,
        "h0_2": heavy.rolling(3).max().values,               # sample day or two days before
        "n7": heavy.rolling(7).sum().values,                 # heavy days in the past week
        "rain3": s.rolling(3).sum().values,                  # mm over three days
        "h_lead": heavy[::-1].rolling(3).max()[::-1].shift(-1).values,  # next three days (placebo)
    }))
rain = pd.concat(rain_rows, ignore_index=True)
pan = obs.merge(rain, on=["point", "date"], how="inner").merge(pts[["point", "region", "name"]], on="point")
pan["logc"] = np.log10(pan["val"].clip(lower=pan.groupby("group")["val"].transform(lambda x: x[x > 0].quantile(0.01))))
pan["exc"] = (pan["val"] > pan.groupby(["point", "group"])["val"].transform(lambda x: x.quantile(0.9))).astype(float)
pan["pm"] = (pan["point"] + "_" + pan["date"].dt.month.astype(str)).astype("category")
pan["rw"] = (pan["region"].astype(str) + "_" + pan["date"].dt.strftime("%G-%V")).astype("category")
pan["n7c"] = pan["n7"].clip(upper=3)
pan["rain10"] = pan["rain3"] / 10
pan.to_csv(OUT / "england_panel.csv.gz", index=False)
print(f"panel: {len(pan):,} samples; heavy-rain share {pan['h0_2'].mean():.3f}", flush=True)

rows = []
for g in ["tss", "bod", "ammonia"]:
    d = pan[pan["group"] == g]
    for name, y, f in [("main", "logc", "h0_2"), ("placebo (next 3 days)", "logc", "h0_2 + h_lead"),
                       ("dose: heavy days in past week", "logc", "n7c"), ("dose: rain over 3 days (per 10 mm)", "logc", "rain10"),
                       ("excursion above works' 90th pct", "exc", "h0_2")]:
        m = pf.feols(f"{y} ~ {f} | pm + rw", data=d, vcov={"CRV1": "point"})
        t = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
        rows.append(t.assign(group=g, spec=name, outcome=y, n=m._N, works=d["point"].nunique(), base=d[y].mean()))
res = pd.concat(rows, ignore_index=True)
res["pct"] = np.where(res["outcome"] == "logc", (10 ** res["Estimate"] - 1) * 100, res["Estimate"] * 100)
res["lo"] = np.where(res["outcome"] == "logc", (10 ** res["2.5%"] - 1) * 100, res["2.5%"] * 100)
res["hi"] = np.where(res["outcome"] == "logc", (10 ** res["97.5%"] - 1) * 100, res["97.5%"] * 100)
res.to_csv(OUT / "england_results.csv", index=False)
pd.set_option("display.width", 220)
print(res[["group", "spec", "term", "pct", "lo", "hi", "Pr(>|t|)", "n", "works"]].round(4).to_string(index=False))
