"""US rain effects by HydroWASTE treatment level and plant size, for transfer to the global plant
set. Strict FE as in the main analysis (plant x calendar month + state x year-month; SEs by
state). Estimated separately in each subsample so each has its own CI.
Output: out/transfer_effects.csv
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
med = link["POP_SERVED"].median()
link["size"] = np.where(link["POP_SERVED"] >= med, "large", "small")
cols = ["permit", "group", "ym", "cal_m", "exceed", "n_heavy", "STATE_CODE"]
p = pd.concat([c[c["group"].isin(["ecoli", "fecal", "tss", "bod"]) & c["permit"].isin(set(link["permit"]))
                 & c["exceed"].notna()]
               for c in pd.read_csv(W / "panel.csv.gz", usecols=cols, chunksize=1_000_000,
                                    dtype={"permit": str, "ym": str, "STATE_CODE": str})])
attr = pd.read_csv(W / "plants_attr.csv", dtype={"permit": str}, usecols=["permit", "cso_city"])
p = p.merge(link[["permit", "LEVEL", "size"]], on="permit").merge(attr, on="permit", how="left")
p["pm"] = (p["permit"] + "_" + p["cal_m"].astype(str)).astype("category")
p["sym"] = (p["STATE_CODE"] + "_" + p["ym"]).astype("category")
p["r1"], p["r2"] = (p["n_heavy"] == 1).astype(float), (p["n_heavy"] >= 2).astype(float)
rows = []
for g in ["ecoli", "fecal", "tss", "bod"]:
    s = p[p["group"] == g]
    subsets = {"all linked": s, "Secondary": s[s["LEVEL"] == "Secondary"], "Advanced": s[s["LEVEL"] == "Advanced"],
               "small (below median population)": s[s["size"] == "small"], "large": s[s["size"] == "large"],
               "Secondary, separate sewers": s[(s["LEVEL"] == "Secondary") & (s["cso_city"] != 1)],
               "Advanced, separate sewers": s[(s["LEVEL"] == "Advanced") & (s["cso_city"] != 1)]}
    for name, d in subsets.items():
        if d["permit"].nunique() < 100:
            continue
        m = pf.feols("exceed ~ r1 + r2 | pm + sym", data=d, vcov={"CRV1": "STATE_CODE"})
        t = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
        rows.append(t.assign(group=g, subset=name, n=m._N, plants=d["permit"].nunique(), base=d["exceed"].mean()))
    print("done", g, flush=True)
res = pd.concat(rows, ignore_index=True)
res.to_csv(G / "out" / "transfer_effects.csv", index=False)
res["pp"], res["lo"], res["hi"] = res["Estimate"] * 100, res["2.5%"] * 100, res["97.5%"] * 100
pd.set_option("display.width", 200)
print(res[["group", "subset", "term", "pp", "lo", "hi", "Pr(>|t|)", "plants", "base"]].round(3).to_string(index=False))
