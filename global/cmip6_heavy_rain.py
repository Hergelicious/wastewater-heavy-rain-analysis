"""Heavy-rain month frequencies at all HydroWASTE plants from NEX-GDDP-CMIP6 (0.25 deg, daily,
bias-corrected). Fifteen models, one per modelling centre; historical 1995-2014 vs 2041-2060 under SSP2-4.5 and SSP5-8.5.
A heavy-rain day = above the cell's own historical 99th percentile of wet-day rain (> 1 mm), as in
the US and England analyses (each model is its own baseline, so model bias cancels).
Output: out/cmip6_shares.csv (cell x model x scenario: share of months with 1 and >=2 heavy days,
heavy days per year), out/hydrowaste_cells.csv (plant -> cell)
"""
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

G = Path(os.environ.get("GLOBAL_DIR", "global"))
TMP = G / "raw" / "cmip6_tmp"
TMP.mkdir(parents=True, exist_ok=True)
OUT = G / "out"
OUT.mkdir(exist_ok=True)
S3 = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com"
MODELS = ["ACCESS-CM2", "GFDL-ESM4", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL",  # first five
          "BCC-CSM2-MR", "CanESM5", "CNRM-CM6-1", "EC-Earth3", "IPSL-CM6A-LR", "MIROC6",
          "NorESM2-MM", "INM-CM5-0", "CMCC-ESM2", "FGOALS-g3"]  # one model per modelling centre
PERIODS = {"historical": range(1995, 2015), "ssp245": range(2041, 2061), "ssp585": range(2041, 2061)}


def keys(model, scen):
    out, token = [], None
    while True:
        url = f"{S3}/?list-type=2&prefix=NEX-GDDP-CMIP6/{model}/{scen}/&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token)
        txt = urllib.request.urlopen(url, timeout=120).read().decode()
        out += [k for k in re.findall(r"<Key>([^<]+)</Key>", txt) if "/pr/" in k]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", txt)
        if not m:
            return out
        token = m.group(1)


def pick(ks, year):
    """Prefer _v2.0, then _v1.1, then the original file for a year."""
    yr = [k for k in ks if re.search(rf"_{year}(_v[\d.]+)?\.nc$", k)]
    for tag in ["_v2.0.nc", "_v1.1.nc", ".nc"]:
        c = [k for k in yr if k.endswith(tag) and (tag != ".nc" or not re.search(r"_v[\d.]+\.nc$", k))]
        if c:
            return c[0]
    return None


def fetch(key):
    f = TMP / Path(key).name
    for k in range(4):
        try:
            urllib.request.urlretrieve(f"{S3}/{key}", f)
            return f
        except Exception as e:
            err = e
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"{key}: {err}")


h = pd.read_csv(G / "hydrowaste" / "HydroWASTE_v10.csv", encoding="latin-1")
lat0, lon0, res = -59.875, 0.125, 0.25
li = np.clip(np.round((h["LAT_OUT"].values - lat0) / res).astype(int), 0, 599)
lo = np.round(((h["LON_OUT"].values % 360) - lon0) / res).astype(int) % 1440
h["cell"] = li * 10000 + lo
h[["WASTE_ID", "cell"]].to_csv(OUT / "hydrowaste_cells.csv", index=False)
cells = np.unique(h["cell"].values)
ci, cj = cells // 10000, cells % 10000
print("cells:", len(cells), flush=True)

rows = []
done_f = OUT / "cmip6_shares.csv"
done = pd.read_csv(done_f) if done_f.exists() else pd.DataFrame(columns=["model"])
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(2)


def read_cells(f):
    with xr.open_dataset(f) as ds:
        nt = ds.sizes["time"]
        a = np.empty((nt, len(ci)), np.float32)
        for t0 in range(0, nt, 92):  # read a quarter-year at a time to keep memory low
            a[t0:t0 + 92] = ds["pr"][t0:t0 + 92].values[:, ci, cj] * 86400.0  # mm/day, days x cells
        t = ds["time"].values
    f.unlink()
    mo = np.array([int(str(x)[5:7]) for x in t])
    return a.astype(np.float32), mo


def stream(model, scen, years):
    """Yield (year, daily cells array, month index) with the next file downloading meanwhile."""
    ks = keys(model, scen)
    todo = [(y, pick(ks, y)) for y in years]
    todo = [(y, k) for y, k in todo if k is not None]
    futs = {}
    for i, (y, k) in enumerate(todo):
        for y2, k2 in todo[i:i + 2]:
            if y2 not in futs:
                futs[y2] = pool.submit(fetch, k2)
        f = futs.pop(y).result()
        a, mo = read_cells(f)
        yield y, a, mo


def monthly_counts(heavy, mo):
    return np.stack([heavy[mo == m].sum(0) for m in range(1, 13)])  # 12 x cells


for model in MODELS:
    if model in set(done["model"]):
        continue
    try:
        t0 = time.time()
        hist = np.empty((366 * len(PERIODS["historical"]), len(cells)), np.float32)  # filled in place
        hmo, off = [], 0
        for y, a, mo in stream(model, "historical", PERIODS["historical"]):
            hist[off:off + len(a)] = a
            off += len(a)
            hmo.append(mo)
            del a
        hist = hist[:off]
        p99 = np.empty(hist.shape[1], np.float32)
        for c0 in range(0, hist.shape[1], 1000):  # chunked to keep memory low
            blk = hist[:, c0:c0 + 1000]
            p99[c0:c0 + 1000] = np.nanpercentile(np.where(blk > 1.0, blk, np.nan), 99, axis=0)
        print(f"  {model} historical {hist.shape} {time.time() - t0:.0f}s", flush=True)
        counts = {"historical": []}
        nyr = len(hmo)
        off = 0
        for mo in hmo:
            n = len(mo)
            counts["historical"].append(monthly_counts(hist[off:off + n] > p99, mo))
            off += n
        del hist
        for scen in ["ssp245", "ssp585"]:
            counts[scen] = [monthly_counts(a > p99, mo) for y, a, mo in stream(model, scen, PERIODS[scen])]
            print(f"  {model} {scen} {time.time() - t0:.0f}s", flush=True)
        for scen, c in counts.items():
            c = np.concatenate(c)  # (years*12) x cells
            rows.append(pd.DataFrame({"cell": cells, "model": model, "scenario": scen,
                                      "share_1": (c == 1).mean(0), "share_2plus": (c >= 2).mean(0),
                                      "heavy_days_per_year": c.sum(0) / (len(c) / 12), "p99_hist_mm": p99,
                                      "years": len(c) / 12}))
        pd.concat([done] + rows, ignore_index=True).to_csv(done_f, index=False)
        print(f"{model} done in {time.time() - t0:.0f}s", flush=True)

    except Exception as exc:  # a model with missing files is skipped, not fatal
        print(f"{model} skipped: {exc}", flush=True)
print("finished", flush=True)
