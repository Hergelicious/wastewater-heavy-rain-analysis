"""Covariance of the one- and >=2-heavy-rain-day effects in the separate-sewer treatment-level subsets
used by the global projection (same models as transfer_effects.py). Output: out/transfer_vcov.csv
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
G = Path(os.environ.get("GLOBAL_DIR", "global"))
W = Path(os.environ.get("US_DIR", "wwtp_national"))
link = pd.read_csv(G / "out" / "link_us.csv", dtype={"permit": str})
link = link[link["LEVEL"].isin(["Secondary", "Advanced"])]
cols = ["permit", "group", "ym", "cal_m", "exceed", "n_heavy", "STATE_CODE"]
p = pd.concat([c[c["group"].isin(["ecoli", "fecal", "tss", "bod"]) & c["permit"].isin(set(link["permit"]))
                 & c["exceed"].notna()]
               for c in pd.read_csv(W / "panel.csv.gz", usecols=cols, chunksize=1_000_000,
                                    dtype={"permit": str, "ym": str, "STATE_CODE": str})])
attr = pd.read_csv(W / "plants_attr.csv", dtype={"permit": str}, usecols=["permit", "cso_city"])
p = p.merge(link[["permit", "LEVEL"]], on="permit").merge(attr, on="permit", how="left")
p["pm"] = (p["permit"] + "_" + p["cal_m"].astype(str)).astype("category")
p["sym"] = (p["STATE_CODE"] + "_" + p["ym"]).astype("category")
p["r1"], p["r2"] = (p["n_heavy"] == 1).astype(float), (p["n_heavy"] >= 2).astype(float)
rows = []
for g in ["ecoli", "fecal", "tss", "bod"]:
    for lvl in ["Secondary", "Advanced"]:
        d = p[(p["group"] == g) & (p["LEVEL"] == lvl) & (p["cso_city"] != 1)]
        m = pf.feols("exceed ~ r1 + r2 | pm + sym", data=d, vcov={"CRV1": "STATE_CODE"})
        b = m.coef()
        v = np.asarray(m._vcov)
        rows.append(dict(group=g, subset=f"{lvl}, separate sewers", b1=b["r1"], b2=b["r2"], var1=v[0, 0], var2=v[1, 1],
                         cov12=v[0, 1], corr=v[0, 1] / np.sqrt(v[0, 0] * v[1, 1]), plants=d["permit"].nunique()))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(G / "out" / "transfer_vcov.csv", index=False)
