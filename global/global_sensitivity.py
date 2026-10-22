"""Sensitivity of the global projection to which US response is transferred. For each pollutant,
the world population-weighted change in rain-driven failures (2041-2060 vs 1995-2014, multi-model
mean shares) is recomputed with the heavy-rain effects (b1, b2) estimated in each US subgroup:
climate terciles (annual rain, mean temperature), census regions, periods, and the treatment-level
effects used in the main projection. Point estimates only (negative effects set to zero).
Output: out/global_sensitivity.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

G = Path(os.environ.get("GLOBAL_DIR", "global"))
US = Path(os.environ.get("US_DIR", "wwtp_national"))
OUT = G / "out"
hw = pd.read_csv(G / "hydrowaste" / "HydroWASTE_v10.csv", encoding="latin-1",
                 usecols=["WASTE_ID", "POP_SERVED", "LEVEL"]).merge(pd.read_csv(OUT / "hydrowaste_cells.csv"),
                                                                    on="WASTE_ID")
pop = hw["POP_SERVED"].fillna(0).to_numpy()
sh = pd.read_csv(OUT / "cmip6_shares.csv")
nmod = sh["model"].nunique()
m = sh.groupby(["cell", "scenario"])[["share_1", "share_2plus"]].mean().unstack("scenario")
x = m.reindex(hw["cell"].values)
tt = pd.concat([pd.read_csv(f) for f in sorted((US / "upgrades").glob("transfer_tests_*.csv"))])
tt = tt[(tt["outcome"] == "exceed") & ~tt["level"].str.startswith("interaction")]
main = pd.read_csv(US / "upgrades" / "model_results.csv")
main = main[(main["model"] == "main") & (main["how"] == "ols") & (main["outcome"] == "exceed")]


def world_change(b1, b2, scen):
    b1, b2 = max(b1, 0.0), max(b2, 0.0)
    t = 12 * (x[("share_1", "historical")] * b1 + x[("share_2plus", "historical")] * b2).to_numpy()
    f = 12 * (x[("share_1", scen)] * b1 + x[("share_2plus", scen)] * b2).to_numpy()
    ok = np.isfinite(t) & np.isfinite(f)
    return 100 * ((f[ok] * pop[ok]).sum() / (t[ok] * pop[ok]).sum() - 1)


rows = []
for g in ["ecoli", "fecal", "tss", "bod"]:
    sources = {("all US plants", "main"): main[main["group"] == g].set_index("term")["Estimate"]}
    for (dim, lv), d in tt[tt["group"] == g].groupby(["dim", "level"]):
        sources[(dim, lv)] = d.set_index("term")["Estimate"]
    for (dim, lv), b in sources.items():
        for scen in ["ssp245", "ssp585"]:
            rows.append(dict(pollutant=g, source_dim=dim, source_level=lv, scenario=scen, b1=b["r1"], b2=b["r2"],
                             world_pct_change=world_change(b["r1"], b["r2"], scen), models=nmod))
out = pd.DataFrame(rows)
out.to_csv(OUT / "global_sensitivity.csv", index=False)
summ = out.groupby(["pollutant", "scenario"])["world_pct_change"].agg(["min", "median", "max"]).round(1)
print(summ.to_string())
print(out[out["scenario"] == "ssp245"].round(2).to_string(index=False))
