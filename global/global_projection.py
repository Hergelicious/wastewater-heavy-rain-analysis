"""Global projection of rain-driven effluent failures at 58,502 HydroWASTE plants.

For each plant: share of months with 1 and >=2 heavy-rain days today (CMIP6 historical 1995-2014)
and in 2041-2060 (SSP2-4.5, SSP5-8.5), 15-model ensemble; rain-driven failure-months per year =
12 x (share_1 x b1 + share_2plus x b2), with b1, b2 the US effects (violation probability, strict
fixed effects) for the plant's treatment level at separate-sewer plants (no global sewer-type
data; combined-sewer amplification would add to this). Primary-only plants use secondary-level
effects (no measured response; flagged). Uncertainty: ND joint draws of (b1, b2) from a bivariate normal with the models'
cluster-robust covariance (out/transfer_vcov.csv), truncated at zero, applied to the multi-model mean; the
single-model range; and a combined interval pooling every model x draw (15 x ND). Default ND = 1,000.
Weights: population served (main; plants without a population count get zero weight), and as
alternatives every plant equally and treated wastewater discharge (out/global_summary_weights.csv).
Groups: world, UN M49 regions and sub-regions, least developed countries, World Bank income group
(Natural Earth attribute), largest countries, primary-only plants, low-dilution plants.
Outputs: out/global_plants.csv, out/global_summary.csv, out/global_exposure.csv
"""
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

G = Path(os.environ.get("GLOBAL_DIR", "global"))
OUT = G / "out"
rng = np.random.default_rng(3)
ND = int(os.environ.get("ND", "1000"))
CLIP = os.environ.get("NOCLIP", "0") != "1"  # NOCLIP=1: keep negative coefficient draws (sensitivity)
TAG = os.environ.get("TAG", "")  # output suffix for sensitivity runs
POLL = {"ecoli": "Pathogens (E. coli)", "fecal": "Pathogens (fecal coliform)", "tss": "Suspended solids",
        "bod": "Organic matter (BOD)"}
INCOME = {"1. High income: OECD": "High income", "2. High income: nonOECD": "High income",
          "3. Upper middle income": "Upper-middle income", "4. Lower middle income": "Lower-middle income",
          "5. Low income": "Low income"}

hw = pd.read_csv(G / "hydrowaste" / "HydroWASTE_v10.csv", encoding="latin-1")
hw = hw.merge(pd.read_csv(OUT / "hydrowaste_cells.csv"), on="WASTE_ID")
cc = pd.read_csv(G / "ext" / "country-codes.csv", dtype=str)
cc = cc[["ISO3166-1-Alpha-3", "Region Name", "Sub-region Name", "Least Developed Countries (LDC)"]] \
    .rename(columns={"ISO3166-1-Alpha-3": "CNTRY_ISO"})
hw = hw.merge(cc, on="CNTRY_ISO", how="left")
ne = gpd.read_file(G / "ext" / "ne" / "ne_50m_admin_0_countries.shp", ignore_geometry=True)
hw = hw.merge(ne[["ADM0_A3", "INCOME_GRP"]].rename(columns={"ADM0_A3": "CNTRY_ISO"}), on="CNTRY_ISO", how="left")
hw["income"] = hw["INCOME_GRP"].map(INCOME)
hw["LDC"] = hw["Least Developed Countries (LDC)"].fillna("").str.lower().eq("x")
hw["level_eff"] = hw["LEVEL"].replace({"Primary": "Secondary"})
hw["low_dilution"] = hw["DF"] < 10
pop = hw["POP_SERVED"].fillna(0).to_numpy()
dis = hw["WASTE_DIS"].fillna(0).clip(lower=0).to_numpy()
COUNTRIES = hw.groupby("CNTRY_ISO")["POP_SERVED"].sum().nlargest(10).index.tolist()  # ten largest by population served

sh = pd.read_csv(OUT / "cmip6_shares.csv")
models = sorted(sh["model"].unique())
print("models:", models, flush=True)
piv = sh.pivot_table(index=["cell", "model"], columns="scenario", values=["share_1", "share_2plus"])
vc = pd.read_csv(OUT / "transfer_vcov.csv")

groups = {"World": np.ones(len(hw), bool)}
for col, pre in [("Region Name", "Region"), ("Sub-region Name", "Sub-region"), ("income", "Income")]:
    for v in sorted(hw[col].dropna().unique()):
        groups[f"{pre}: {v}"] = (hw[col] == v).to_numpy()
groups["Least developed countries"] = hw["LDC"].to_numpy()
for c in COUNTRIES:
    groups[f"Country: {c}"] = (hw["CNTRY_ISO"] == c).to_numpy()
groups["Primary treatment only"] = (hw["LEVEL"] == "Primary").to_numpy()
groups["Secondary treatment"] = (hw["LEVEL"] == "Secondary").to_numpy()
groups["Advanced treatment"] = (hw["LEVEL"] == "Advanced").to_numpy()
groups["River dilution < 10x"] = hw["low_dilution"].to_numpy()


def pct_stats(t, f):
    ok = t > 0
    pct = np.where(ok, 100 * (f / np.where(ok, t, 1) - 1), np.nan)
    return np.nanmedian(pct), np.nanquantile(pct, 0.025), np.nanquantile(pct, 0.975)


plant_rows, summ, expo, wrows = [], [], [], []
for g, label in POLL.items():
    e = {}
    for lvl in ["Secondary", "Advanced"]:
        r = vc[(vc["group"] == g) & (vc["subset"] == f"{lvl}, separate sewers")].iloc[0]
        mu = np.array([r["b1"], r["b2"]])
        cov = np.array([[r["var1"], r["cov12"]], [r["cov12"], r["var2"]]])
        dr = rng.multivariate_normal(mu, cov, size=ND)
        e[lvl] = (np.clip(dr, 0, None) if CLIP else dr).astype(np.float32)  # truncated at 0 unless NOCLIP=1
        e[lvl + "_mean"] = np.clip(mu, 0, None)
    for scen in ["ssp245", "ssp585"]:
        names = list(groups)
        # plants with a defined baseline in every model
        fin = np.ones(len(hw), bool)
        for mdl in models:
            x = piv.xs(mdl, level="model").reindex(hw["cell"].values)
            fin &= np.isfinite(x[[("share_1", "historical"), ("share_2plus", "historical"),
                                  ("share_1", scen), ("share_2plus", scen)]].to_numpy()).all(1)
        WTS = {"population": pop, "plants": np.ones(len(hw)), "discharge": dis}
        wmats = {k: np.stack([np.where(groups[n] & fin, w, 0) for n in names]).astype(np.float32) for k, w in WTS.items()}
        Tm = {k: [] for k in WTS}
        Fm = {k: [] for k in WTS}
        per_model = {}  # model -> (today, future) at point-estimate coefficients
        for mdl in models:
            x = piv.xs(mdl, level="model").reindex(hw["cell"].values)
            s1h = np.nan_to_num(x[("share_1", "historical")].to_numpy(np.float32))
            s2h = np.nan_to_num(x[("share_2plus", "historical")].to_numpy(np.float32))
            s1f = np.nan_to_num(x[("share_1", scen)].to_numpy(np.float32))
            s2f = np.nan_to_num(x[("share_2plus", scen)].to_numpy(np.float32))
            today = np.zeros((len(hw), ND), np.float32)
            fut = np.zeros((len(hw), ND), np.float32)
            pt, pf_ = np.full(len(hw), np.nan), np.full(len(hw), np.nan)
            for lvl in ["Secondary", "Advanced"]:
                m = (hw["level_eff"] == lvl).to_numpy() & fin
                b = e[lvl]
                today[m] = 12 * (s1h[m, None] * b[None, :, 0] + s2h[m, None] * b[None, :, 1])
                fut[m] = 12 * (s1f[m, None] * b[None, :, 0] + s2f[m, None] * b[None, :, 1])
                bm = e[lvl + "_mean"]
                pt[m] = 12 * (s1h[m] * bm[0] + s2h[m] * bm[1])
                pf_[m] = 12 * (s1f[m] * bm[0] + s2f[m] * bm[1])
            per_model[mdl] = (pt, pf_)
            for k, wm in wmats.items():
                Tm[k].append((wm @ today).astype(np.float64))
                Fm[k].append((wm @ fut).astype(np.float64))
            del today, fut
        for k in WTS:
            Tk, Fk = np.stack(Tm[k]), np.stack(Fm[k])  # models x groups x draws
            T, F = Tk.mean(0), Fk.mean(0)  # multi-model mean, per draw
            pooled = 100 * (Fk / np.where(Tk > 0, Tk, np.nan) - 1)  # every model x draw
            w = WTS[k]
            for gi, name in enumerate(names):
                ok = groups[name] & fin
                if ok.sum() == 0 or wmats[k][gi].sum() == 0:
                    continue
                med, lo, hi = pct_stats(T[gi], F[gi])
                mp = [100 * (np.nansum(a2[ok] * w[ok]) / np.nansum(a1[ok] * w[ok]) - 1) for a1, a2 in per_model.values()]
                row = dict(pollutant=g, label=label, scenario=scen, group=name, plants=int(ok.sum()),
                           pop_served_m=pop[ok].sum() / 1e6, pct_change_popweighted=med, lo=lo, hi=hi,
                           model_min=np.min(mp), model_max=np.max(mp), models_increase=int(np.sum(np.array(mp) > 0)),
                           comb_lo=np.nanquantile(pooled[:, gi, :], 0.025), comb_hi=np.nanquantile(pooled[:, gi, :], 0.975))
                if k == "population":
                    summ.append(row)
                elif name == "World" or name.startswith("Income") or name.startswith("Region") or name == "Least developed countries":
                    wrows.append(dict(row, weight=k))
        # plant-level change (multi-model mean, point coefficients) and model agreement
        pt_all = np.mean([v[0] for v in per_model.values()], 0)
        pf_all = np.mean([v[1] for v in per_model.values()], 0)
        chg = np.where(pt_all > 0, 100 * (pf_all / np.where(pt_all > 0, pt_all, 1) - 1), np.nan)
        agree = np.sum([v[1] > v[0] for v in per_model.values()], 0)
        plant_rows.append(pd.DataFrame({"WASTE_ID": hw["WASTE_ID"], "pollutant": g, "scenario": scen,
                                        "today": pt_all, "future": pf_all, "pct_change": chg,
                                        "models_increase": agree}))
        for name in ["World", "Least developed countries", "Income: Low income", "Income: Lower-middle income",
                     "Income: Upper-middle income", "Income: High income"]:
            m = groups[name] & np.isfinite(chg)
            w = pop[m].sum()
            expo.append(dict(pollutant=g, scenario=scen, group=name, pop_served_m=w / 1e6,
                             share_pop_increase=pop[m & (chg > 0)].sum() / w,
                             share_pop_gt25=pop[m & (chg > 25)].sum() / w,
                             share_pop_gt50=pop[m & (chg > 50)].sum() / w,
                             share_pop_gt100=pop[m & (chg > 100)].sum() / w,
                             share_pop_80pct_models_agree=pop[m & (agree >= np.ceil(0.8 * len(models)))].sum() / w))
        print("done", g, scen, flush=True)

pd.concat(plant_rows).to_csv(OUT / f"global_plants{TAG}.csv", index=False)
s = pd.DataFrame(summ)
s.to_csv(OUT / f"global_summary{TAG}.csv", index=False)
x = pd.DataFrame(expo)
x.to_csv(OUT / f"global_exposure{TAG}.csv", index=False)
pd.DataFrame(wrows).to_csv(OUT / f"global_summary_weights{TAG}.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
print(s[s["pollutant"] == "tss"].drop(columns=["label"]).round(1).to_string(index=False))
print(x.round(3).to_string(index=False))
