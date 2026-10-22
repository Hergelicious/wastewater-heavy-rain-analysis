"""Dilution and downstream reach for each municipal plant, following the HydroWASTE method:
outfall snapped to the nearest HydroRIVERS reach (within 2 km, else 10 km); dilution factor =
long-term mean river discharge / plant effluent flow; then downstream routing along NEXT_DOWN.

Inputs: plants.csv (pipeline), ICIS-NPDES outfall coordinates, NPDES catchment indexing
(coastal/tidal flags), HydroRIVERS v1.0 North America.  Output: plants_dilution.csv
"""
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
RAW = W / "raw"
RIV = Path("HydroRIVERS_v10_na_shp/HydroRIVERS_v10_na.shp")
CRS = "ESRI:102008"  # North America Albers equal-area, metres
MGD_TO_CMS = 0.0438126

plants = pd.read_csv(W / "plants.csv", dtype={"permit": str})
keep = set((W / "dmr_permits.txt").read_text().split())
plants = plants[plants["permit"].isin(keep)].copy()

# main outfall coordinates where available (feature 001), else facility coordinates
with zipfile.ZipFile(RAW / "npdes_downloads.zip") as z:
    oc = pd.read_csv(z.open("NPDES_PERM_FEATURE_COORDS.csv"), dtype=str, encoding="latin-1")
oc.columns = oc.columns.str.upper()
idcol = "EXTERNAL_PERMIT_NMBR" if "EXTERNAL_PERMIT_NMBR" in oc else "NPDES_ID"
oc["lat_o"] = pd.to_numeric(oc["LATITUDE_MEASURE"], errors="coerce")
oc["lon_o"] = pd.to_numeric(oc["LONGITUDE_MEASURE"], errors="coerce")
oc = oc[oc["lat_o"].between(17, 72) & oc["lon_o"].between(-180, -60)]
oc["is001"] = oc["PERM_FEATURE_NMBR"].astype(str).str.strip().isin(["001", "1", "0001"])
oc = oc.sort_values("is001", ascending=False).drop_duplicates(idcol)[[idcol, "lat_o", "lon_o"]]
plants = plants.merge(oc.rename(columns={idcol: "permit"}), on="permit", how="left")
plants["outfall_coords"] = plants["lat_o"].notna()
plants["y"] = plants["lat_o"].fillna(plants["lat"])
plants["x"] = plants["lon_o"].fillna(plants["lon"])

with zipfile.ZipFile(RAW / "npdes_attains_downloads.zip") as z:
    cat = pd.read_csv(z.open("NPDES_CATCHMENTS.csv"), dtype=str, encoding="latin-1",
                      usecols=["NPDES_ID", "SUB_ID", "COASTAL", "TIDAL"])
cat = cat[cat["SUB_ID"].fillna("").str.upper().eq("BESTPICK")].drop_duplicates("NPDES_ID")
plants = plants.merge(cat.rename(columns={"NPDES_ID": "permit"})[["permit", "COASTAL", "TIDAL"]],
                      on="permit", how="left")
plants["coastal"] = (plants["COASTAL"].eq("Y") | plants["TIDAL"].eq("Y")).astype(int)

eff = plants["actual_mgd"].where(plants["actual_mgd"] > 0, plants["design_mgd"] * 0.7)
plants["eff_cms"] = eff * MGD_TO_CMS

print("loading HydroRIVERS ...", flush=True)
riv = gpd.read_file(RIV, columns=["HYRIV_ID", "NEXT_DOWN", "LENGTH_KM", "DIS_AV_CMS", "UPLAND_SKM"],
                    engine="pyogrio").to_crs(CRS)
pts = gpd.GeoDataFrame(plants[["permit"]], geometry=gpd.points_from_xy(plants["x"], plants["y"]),
                       crs="EPSG:4326").to_crs(CRS)
snap = gpd.sjoin_nearest(pts, riv[["HYRIV_ID", "DIS_AV_CMS", "geometry"]], how="left",
                         max_distance=10_000, distance_col="snap_m")
snap = snap.sort_values("snap_m").drop_duplicates("permit")[["permit", "HYRIV_ID", "DIS_AV_CMS", "snap_m"]]
plants = plants.merge(snap, on="permit", how="left")

# downstream routing
nxt = dict(zip(riv["HYRIV_ID"], riv["NEXT_DOWN"]))
length = dict(zip(riv["HYRIV_ID"], riv["LENGTH_KM"]))
dis = dict(zip(riv["HYRIV_ID"], riv["DIS_AV_CMS"]))

# cumulative effluent from all upstream plants on each reach (HydroWASTE "wastewater ratio")
cum = {}
for rid, q in zip(plants["HYRIV_ID"], plants["eff_cms"]):
    if pd.isna(rid) or pd.isna(q):
        continue
    r = int(rid)
    while r and r in nxt:
        cum[r] = cum.get(r, 0.0) + q
        r = int(nxt[r])

dn_km, ww_share = [], []
for rid, q in zip(plants["HYRIV_ID"], plants["eff_cms"]):
    if pd.isna(rid) or pd.isna(q) or q <= 0:
        dn_km.append(np.nan), ww_share.append(np.nan)
        continue
    r, km = int(rid), 0.0
    ww_share.append(cum.get(r, q) / max(dis.get(r, 0.0), 1e-6))
    while r and r in nxt and dis[r] / q < 100:  # until this plant's effluent is < 1 % of flow
        km += length[r]
        r = int(nxt[r])
    dn_km.append(km)
plants["dn_km_df100"] = dn_km
plants["ww_ratio_at_outfall"] = ww_share
plants["dilution_factor"] = plants["DIS_AV_CMS"] / plants["eff_cms"].where(plants["eff_cms"] > 0)
plants.loc[plants["coastal"] == 1, ["dilution_factor", "dn_km_df100", "ww_ratio_at_outfall"]] = np.nan

out = plants[["permit", "x", "y", "outfall_coords", "coastal", "eff_cms", "HYRIV_ID", "DIS_AV_CMS",
              "snap_m", "dilution_factor", "dn_km_df100", "ww_ratio_at_outfall"]]
out.to_csv(W / "plants_dilution.csv", index=False)
inl = out[(out["coastal"] == 0) & out["dilution_factor"].notna()]
print(f"plants: {len(out):,}; outfall coords {out['outfall_coords'].mean():.0%}; coastal/tidal "
      f"{out['coastal'].sum():,}; snapped inland {len(inl):,} (median snap {inl['snap_m'].median():.0f} m)")
print("dilution factor quantiles:", inl["dilution_factor"].quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
print(f"share with DF < 10: {(inl['dilution_factor'] < 10).mean():.1%}; median km to DF100: "
      f"{inl['dn_km_df100'].median():.1f}")
