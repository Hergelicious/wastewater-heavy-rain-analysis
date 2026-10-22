"""Headline numbers for the global outlook: change in heavy-rain days at plants (multi-model mean),
population served at plants where they increase, and model agreement. Output: out/global_stats.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

G = Path(os.environ.get("GLOBAL_DIR", "global"))
OUT = G / "out"
hw = pd.read_csv(G / "hydrowaste" / "HydroWASTE_v10.csv", encoding="latin-1",
                 usecols=["WASTE_ID", "POP_SERVED", "CNTRY_ISO", "LEVEL"]).merge(
    pd.read_csv(OUT / "hydrowaste_cells.csv"), on="WASTE_ID")
sh = pd.read_csv(OUT / "cmip6_shares.csv")
pop = hw["POP_SERVED"].fillna(0)
rows = [dict(stat="plants", value=len(hw)), dict(stat="pop_served_bn", value=pop.sum() / 1e9),
        dict(stat="models", value=sh["model"].nunique())]
hd = sh.pivot_table(index=["cell", "model"], columns="scenario", values="heavy_days_per_year")
rows.append(dict(stat="heavy_days_hist_median", value=hd["historical"].groupby("cell").mean()
                 .reindex(hw["cell"]).median()))
for scen in ["ssp245", "ssp585"]:
    chg = 100 * (hd[scen] / hd["historical"] - 1)
    chg = chg.replace([np.inf, -np.inf], np.nan)
    c_mean = chg.groupby("cell").mean().reindex(hw["cell"]).to_numpy()
    agree = (hd[scen] > hd["historical"]).groupby("cell").sum().reindex(hw["cell"]).to_numpy()
    ok = np.isfinite(c_mean)
    w = pop.to_numpy()
    rows += [dict(stat=f"{scen}_heavy_days_pct_median_plant", value=np.nanmedian(c_mean)),
             dict(stat=f"{scen}_heavy_days_pct_popweighted",
                  value=100 * (np.nansum(w * hd[scen].groupby("cell").mean().reindex(hw["cell"]).to_numpy())
                               / np.nansum(w * hd["historical"].groupby("cell").mean().reindex(hw["cell"]).to_numpy()) - 1)),
             dict(stat=f"{scen}_share_pop_heavy_days_increase", value=w[ok & (c_mean > 0)].sum() / w[ok].sum()),
             dict(stat=f"{scen}_share_pop_80pct_models_increase", value=w[agree >= np.ceil(0.8 * sh["model"].nunique())].sum() / w.sum()),
             dict(stat=f"{scen}_share_plants_increase", value=np.mean(c_mean[ok] > 0))]
    for col in ["share_1", "share_2plus"]:
        p = sh.pivot_table(index=["cell", "model"], columns="scenario", values=col)
        t = (p["historical"].groupby("cell").mean().reindex(hw["cell"]).to_numpy() * w)
        f = (p[scen].groupby("cell").mean().reindex(hw["cell"]).to_numpy() * w)
        rows.append(dict(stat=f"{scen}_{col}_pct_popweighted", value=100 * (np.nansum(f) / np.nansum(t) - 1)))
out = pd.DataFrame(rows)
out.to_csv(OUT / "global_stats.csv", index=False)
print(out.to_string(index=False))
