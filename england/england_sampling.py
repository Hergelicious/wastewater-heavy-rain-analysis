"""Does sampling in England change around heavy rain? (selective-sampling check)
For each works, every day between its first and last final-effluent sample in 2016-2025 is a works-day;
a sampling occasion is a works-day with at least one BOD, ammonia or suspended-solids measurement.
(1) Descriptive: share of works-days vs share of sampling occasions that fall on a heavy-rain window
    (heavy rain on the day or the two days before; works' own 99th percentile of wet-day rain).
(2) Linear probability model on works-days: sampled ~ heavy-rain window + placebo (heavy rain in the next
    three days), with works x calendar-month and region x ISO-week fixed effects; also day-of-week
    fixed effects; SEs clustered by works.
Inputs: out/england_panel.csv.gz, raw/ea_points.csv, E-OBS netCDF. Output: out/england_sampling.csv
"""
import os
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
pan = pd.read_csv(OUT / "england_panel.csv.gz", usecols=["point", "group", "date", "region", "purpose"], parse_dates=["date"])
n_meas = len(pan)
occ = pan.drop_duplicates(["point", "date"])[["point", "date", "region"]]
print(f"measurements {n_meas:,}; sampling occasions {len(occ):,}; works {occ['point'].nunique():,}", flush=True)

pts = pd.read_csv(RAW / "ea_points.csv")
tr = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
pts = pts[pts["point"].isin(occ["point"].unique())].dropna(subset=["easting", "northing"]).copy()
pts["lon"], pts["lat"] = tr.transform(pts["easting"].astype(float).values, pts["northing"].astype(float).values)
ds = xr.open_dataset(RAW / "rr_ens_mean_0.1deg_reg_2011-2025_v33.0e.nc")
sub = ds["rr"].sel(latitude=slice(49.5, 56.5), longitude=slice(-7, 2.5))
rr = sub.values
lats, lons, times = sub.latitude.values, sub.longitude.values, pd.to_datetime(sub.time.values)
valid = ~np.isnan(rr).all(0)
vi, vj = np.where(valid)
span = occ.groupby("point")["date"].agg(["min", "max"])
reg = occ.drop_duplicates("point").set_index("point")["region"]
sampled = set(zip(occ["point"], occ["date"]))
rows = []
for p in pts.itertuples():
    i, j = np.abs(lats - p.lat).argmin(), np.abs(lons - p.lon).argmin()
    if not valid[i, j]:
        k = np.argmin((vi - i) ** 2 + (vj - j) ** 2)
        i, j = vi[k], vj[k]
    s = pd.Series(rr[:, i, j], index=times).fillna(0)
    p99 = s[s > 0.1].quantile(0.99)
    heavy = (s > p99).astype(np.int8)
    h0_2 = heavy.rolling(3).max()
    h_lead = heavy[::-1].rolling(3).max()[::-1].shift(-1)
    lo, hi = span.loc[p.point]
    idx = (times >= lo) & (times <= hi)
    d = pd.DataFrame({"date": times[idx], "h0_2": h0_2.values[idx], "h_lead": h_lead.values[idx]})
    d["point"] = p.point
    rows.append(d)
wd = pd.concat(rows, ignore_index=True).dropna()
del rows
wd["sampled"] = [(a, b) in sampled for a, b in zip(wd["point"], wd["date"])]
wd["sampled"] = wd["sampled"].astype(np.float32)
wd["region"] = wd["point"].map(reg)
n_wd = len(wd)
print(f"works-days {n_wd:,}", flush=True)

out = []
share_days = wd["h0_2"].mean()
share_occ = wd.loc[wd["sampled"] == 1, "h0_2"].mean()
out.append(dict(stat="share of works-days in a heavy-rain window", value=share_days))
out.append(dict(stat="share of sampling occasions in a heavy-rain window", value=share_occ))
out.append(dict(stat="ratio (occasions / days)", value=share_occ / share_days))
out.append(dict(stat="sampling rate, works-days without heavy-rain window", value=wd.loc[wd["h0_2"] == 0, "sampled"].mean()))
out.append(dict(stat="sampling rate, works-days with heavy-rain window", value=wd.loc[wd["h0_2"] == 1, "sampled"].mean()))
# regression on works-weeks (daily panel too large for memory): sampling occasions in the week vs
# heavy-rain window days in the week and placebo days (heavy rain in the next three days)
wd["wk"] = wd["date"].dt.strftime("%G-%V")
wd["month"] = wd["date"].dt.month
wk = wd.groupby(["point", "wk"]).agg(sampled=("sampled", "sum"), h0_2=("h0_2", "sum"), h_lead=("h_lead", "sum"),
                                     days=("sampled", "size"), month=("month", "first"), region=("region", "first")).reset_index()
del wd
wk = wk[wk["days"] == 7]  # complete weeks only
wk["pm"] = (wk["point"] + "_" + wk["month"].astype(str)).astype("category")
wk["rw"] = (wk["region"].astype(str) + "_" + wk["wk"]).astype("category")
m = pf.feols("sampled ~ h0_2 + h_lead | pm + rw", data=wk, vcov={"CRV1": "point"})
t = m.tidy().reset_index()
base = wk["sampled"].mean()
for _, r in t.iterrows():
    out.append(dict(stat=f"works-week model: {r['Coefficient']} (sampling occasions per heavy-rain-window day)", value=r["Estimate"],
                    lo=r["2.5%"], hi=r["97.5%"], p=r["Pr(>|t|)"], base_per_week=base,
                    relative_pct_per_day=100 * r["Estimate"] / (base / 7)))
out.append(dict(stat="works-weeks", value=m._N))
out.append(dict(stat="measurements", value=n_meas))
out.append(dict(stat="sampling occasions", value=len(occ)))
out.append(dict(stat="works-days", value=n_wd))
res = pd.DataFrame(out)
res.to_csv(OUT / "england_sampling.csv", index=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
