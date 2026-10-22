"""Link HydroWASTE plants to US NPDES permits and English EA final-effluent sampling points by
location, so the rain response can be estimated by HydroWASTE treatment level and size.
Rule: nearest plant within 1 km (US: facility coordinates; England: sampling point), one-to-one
(closest pair kept). Output: out/link_us.csv, out/link_england.csv
"""
from pathlib import Path

import os
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

G = Path(os.environ.get("GLOBAL_DIR", "global"))
OUT = G / "out"
hw = pd.read_csv(G / "hydrowaste" / "HydroWASTE_v10.csv", encoding="latin-1")
tr = Transformer.from_crs("EPSG:4326", "ESRI:102008", always_xy=True)  # metres


def link(a, b, max_m=1000):
    """a, b: DataFrames with lon, lat; returns pairs (ia, ib, dist) one-to-one within max_m."""
    ax, ay = tr.transform(a["lon"].values, a["lat"].values)
    bx, by = tr.transform(b["lon"].values, b["lat"].values)
    d, j = cKDTree(np.c_[bx, by]).query(np.c_[ax, ay], k=1, distance_upper_bound=max_m)
    m = pd.DataFrame({"ia": np.arange(len(a)), "ib": j, "dist_m": d})
    m = m[np.isfinite(m["dist_m"])].sort_values("dist_m").drop_duplicates("ib")
    return m


# --- United States
us = pd.read_csv(Path(os.environ.get("US_DIR", "wwtp_national")) / "plants.csv", dtype={"permit": str}).dropna(subset=["lat", "lon"])
keep = set((Path(os.environ.get("US_DIR", "wwtp_national")) / "dmr_permits.txt").read_text().split())
us = us[us["permit"].isin(keep)].reset_index(drop=True)
hu = hw[hw["SOURCE"] == 2].rename(columns={"LAT_WWTP": "lat", "LON_WWTP": "lon"}).reset_index(drop=True)
m = link(us, hu)
lu = pd.DataFrame({"permit": us.loc[m["ia"], "permit"].values, "WASTE_ID": hu.loc[m["ib"], "WASTE_ID"].values,
                   "dist_m": m["dist_m"].values, "LEVEL": hu.loc[m["ib"], "LEVEL"].values,
                   "POP_SERVED": hu.loc[m["ib"], "POP_SERVED"].values,
                   "hw_dis_m3d": hu.loc[m["ib"], "WASTE_DIS"].values,
                   "npdes_mgd": us.loc[m["ia"], "actual_mgd"].values})
r = (lu["hw_dis_m3d"] / (lu["npdes_mgd"] * 3785.41))
lu["flow_ratio"] = r
lu = lu[~(r.notna() & ((r > 5) | (r < 0.2)))]  # drop pairs whose flows disagree by >5x
lu.to_csv(OUT / "link_us.csv", index=False)
print(f"US: {len(us):,} DMR plants; linked {len(lu):,} ({len(lu) / len(us):.0%}); levels:",
      lu["LEVEL"].value_counts().to_dict())

# --- England
ea = pd.read_csv(G / "raw" / "ea_points.csv").dropna(subset=["easting", "northing"]).reset_index(drop=True)
tb = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
ea["lon"], ea["lat"] = tb.transform(ea["easting"].astype(float).values, ea["northing"].astype(float).values)
hg = hw[hw["CNTRY_ISO"] == "GBR"].rename(columns={"LAT_WWTP": "lat", "LON_WWTP": "lon"}).reset_index(drop=True)
m = link(ea, hg)
le = pd.DataFrame({"point": ea.loc[m["ia"], "point"].values, "WASTE_ID": hg.loc[m["ib"], "WASTE_ID"].values,
                   "dist_m": m["dist_m"].values, "LEVEL": hg.loc[m["ib"], "LEVEL"].values,
                   "POP_SERVED": hg.loc[m["ib"], "POP_SERVED"].values})
le.to_csv(OUT / "link_england.csv", index=False)
print(f"England: {len(ea):,} EA points; linked {len(le):,}; levels:", le["LEVEL"].value_counts().to_dict())
