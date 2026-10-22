"""Heavy-rain-month frequencies under warming (Clausius-Clapeyron scaling of observed rain).

For each plant, daily Daymet precipitation (the same record used in the panel) is scaled by
1.07**dT for local warming dT in {0, 1, 2, 3} degC; days above the plant's historical 99th
percentile wet-day amount are counted per month. Output: share of months with 1 and >=2
heavy-rain days, per plant and calendar month -> projection_shares.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
CC = 0.07
DTS = [0, 1, 2, 3]


def one(path):
    d = pd.read_csv(path)
    d = d[(d["year"] >= 2015) & (d["year"] <= 2025)]
    p99 = d.loc[d["prcp"] > 0.1, "prcp"].quantile(0.99)
    date = pd.to_datetime(d["year"].astype(str) + "-01-01") + pd.to_timedelta(d["yday"] - 1, "D")
    ym = date.dt.to_period("M")
    out = []
    for dt in DTS:
        n = (d["prcp"] * (1 + CC) ** dt > p99).groupby(ym).sum()
        cal = n.index.month
        df = pd.DataFrame({"cal_m": cal, "r1": (n.values == 1), "r2": (n.values >= 2)})
        g = df.groupby("cal_m")[["r1", "r2"]].mean().reset_index()
        g["dT"] = dt
        out.append(g)
    o = pd.concat(out)
    o["permit"] = path.name.replace(".csv.gz", "")
    return o


if __name__ == "__main__":
    keep = set((W / "dmr_permits.txt").read_text().split())
    files = [f for f in (W / "daymet").glob("*.csv.gz") if f.name.replace(".csv.gz", "") in keep]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(one, files, chunksize=50))
    out = pd.concat(res, ignore_index=True)
    out.to_csv(W / "upgrades" / "projection_shares.csv.gz", index=False)
    m = out.groupby("dT")[["r1", "r2"]].mean()
    print(m.round(4))
