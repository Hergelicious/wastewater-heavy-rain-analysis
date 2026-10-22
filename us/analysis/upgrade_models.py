"""Upgrade analyses (pyfixest). Results go to <workdir>/upgrades/.

Main specification: outcome ~ heavy-rain-day bins + temperature-anomaly bins
                    | plant x calendar-month FE + state x year-month FE, SEs clustered by state.
  * exceedance: linear probability (pp) and Poisson (relative risk)
  * concentration: log10 monthly average and log10 monthly peak (daily maximum)
Heterogeneity / policy: combined sewers (CSO city, active and closed outfalls, federal decrees),
receiving-water uses, dilution, nearby communities.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
OUT = W / "upgrades"
OUT.mkdir(exist_ok=True)
ALL_GROUPS = ["ecoli", "fecal", "tss", "bod", "ammonia", "tp"]
MODE = sys.argv[2] if len(sys.argv) > 2 else "attributable"   # a group name, or "attributable"
GROUPS = [MODE] if MODE in ALL_GROUPS else ALL_GROUPS
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


cols = ["permit", "group", "ym", "cal_m", "year", "season", "logc", "logc_max", "exceed",
        "n_heavy", "t_anom", "prcp_max1d", "STATE_CODE"]
parts = []
for ch in pd.read_csv(W / "panel.csv.gz", dtype={"permit": str, "ym": str, "STATE_CODE": str},
                      usecols=cols, chunksize=1_000_000):
    ch = ch[ch["group"].isin(GROUPS)]
    if MODE == "attributable":
        ch = ch[ch["exceed"].notna() & ch["group"].isin(["ecoli", "fecal", "tss", "bod"])]
    for c in ["logc", "logc_max", "exceed", "t_anom", "prcp_max1d"]:
        ch[c] = ch[c].astype("float32")
    parts.append(ch)
p = pd.concat(parts, ignore_index=True)
del parts
acols = ["permit", "cso_city", "cso_active_city", "cso_share_closed", "cso_decree", "dilution_factor",
         "lowincome_share", "rw_recreation_use", "rw_drinking_use", "rw_pathogen_impaired"]
attr = pd.read_csv(W / "plants_attr.csv", dtype={"permit": str}, usecols=acols)
p = p.merge(attr, on="permit", how="left")
p["pm"] = (p["permit"] + "_" + p["cal_m"].astype(str)).astype("category")
p["sym"] = (p["STATE_CODE"].astype(str) + "_" + p["ym"]).astype("category")
for c in ["group", "season", "STATE_CODE"]:
    p[c] = p[c].astype("category")
p["r1"] = (p["n_heavy"] == 1).astype(float)
p["r2"] = (p["n_heavy"] >= 2).astype(float)
edges = [-np.inf, -3, -1.5, 1.5, 3, np.inf]
tb = pd.cut(p["t_anom"], edges, labels=["tm3", "tm15", "t0", "tp15", "tp3"], right=False)
for l in ["tm3", "tm15", "tp15", "tp3"]:
    p[l] = (tb == l).astype(float)
TB = "tm3 + tm15 + tp15 + tp3"
p["winter_t"] = p["t_anom"] * (p["season"] == "DJF")
p["summer_t"] = p["t_anom"] * (p["season"] == "JJA")
p["cso"] = p["cso_city"].fillna(0)
p["log_active"] = np.log1p(p["cso_active_city"].fillna(0))
p["closed_share"] = p["cso_share_closed"].fillna(0)
p["decree"] = p["cso_decree"].fillna(0)
p["low_dil"] = (p["dilution_factor"] < 10).astype(float).where(p["dilution_factor"].notna())
lo_t = p.groupby("permit")["lowincome_share"].first().quantile(2 / 3)
p["low_income_hi"] = (p["lowincome_share"] >= lo_t).astype(float).where(p["lowincome_share"].notna())
log(f"panel {len(p):,} rows")

rows = []


def fit(name, g, d, fml, y, how="ols", vcov=None):
    vcov = vcov or {"CRV1": "STATE_CODE"}
    try:
        if how == "pois":
            # Poisson: plant + state x year-month FE (plant x calendar-month is too sparse for a
            # binary outcome); drop groups that never violate, which carry no information
            fml = fml.replace("| pm + sym", "| permit + sym")
            d = d[d[y].notna()]
            for _ in range(10):
                k = (d.groupby("permit")[y].transform("sum") > 0) & (d.groupby("sym")[y].transform("sum") > 0)
                if k.all():
                    break
                d = d[k]
            m = pf.fepois(f"{y} ~ {fml}", data=d, vcov=vcov, fixef_maxiter=20000)
        else:
            m = pf.feols(f"{y} ~ {fml}", data=d, vcov=vcov)
        t = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
        t = t.assign(model=name, group=g, outcome=y, how=how, n=m._N, plants=d.loc[d[y].notna(), "permit"].nunique(),
                     base=d[y].mean())
        rows.append(t)
    except Exception as e:  # keep going, record the failure
        log(f"  FAILED {name} {g} {y}: {e}")


FE = "| pm + sym"
import gc
for g in (GROUPS if MODE in ALL_GROUPS else []):
    gc.collect()
    s = p[p["group"] == g]
    se = s[s["exceed"].notna()]
    sc = s[s["logc"].notna()]
    sm = s[s["logc_max"].notna()]
    # 1. main effects, strict FE
    fit("main", g, se, f"r1 + r2 + {TB} {FE}", "exceed")
    fit("main", g, se, f"r1 + r2 + {TB} {FE}", "exceed", how="pois")
    fit("main", g, sc, f"r1 + r2 + {TB} {FE}", "logc")
    fit("main", g, sm, f"r1 + r2 + {TB} {FE}", "logc_max")
    fit("main_plantcluster", g, se, f"r1 + r2 + {TB} {FE}", "exceed", vcov={"CRV1": "permit"})
    fit("seasonal_temp", g, se, f"winter_t + summer_t + r1 + r2 {FE}", "exceed")
    fit("seasonal_temp", g, sc, f"winter_t + summer_t + r1 + r2 {FE}", "logc")
    fit("seasonal_temp", g, sm, f"winter_t + summer_t + r1 + r2 {FE}", "logc_max")
    # storm intensity: wettest day of the month (per 10 mm), continuous
    s2 = se.assign(pmax10=se["prcp_max1d"] / 10)
    fit("storm_intensity", g, s2, f"pmax10 + {TB} {FE}", "exceed")
    # 2. combined sewers
    fit("cso", g, se, f"r1 + r2 + r1:cso + r2:cso + {TB} {FE}", "exceed")
    fit("cso", g, se, f"r1 + r2 + r1:cso + r2:cso + {TB} {FE}", "exceed", how="pois")
    fit("cso", g, sm, f"r1 + r2 + r1:cso + r2:cso + {TB} {FE}", "logc_max")
    sc_c = se[se["cso"] == 1]
    if sc_c["permit"].nunique() > 50:
        fit("cso_active_outfalls", g, sc_c, f"r1 + r2 + r1:log_active + r2:log_active + {TB} {FE}", "exceed")
        fit("cso_share_closed", g, sc_c, f"r1 + r2 + r1:closed_share + r2:closed_share + {TB} {FE}", "exceed")
        fit("cso_decree", g, sc_c, f"r1 + r2 + r1:decree + r2:decree + {TB} {FE}", "exceed")
        # within-plant change in rain sensitivity over the decade, decree vs other CSO plants
        sc_c = sc_c.assign(trend=(sc_c["year"] - 2020) / 5)
        fit("cso_decree_trend", g, sc_c,
            f"r1 + r2 + r1:trend + r2:trend + r1:decree + r2:decree + r1:trend:decree + r2:trend:decree + {TB} {FE}",
            "exceed")
    # 3. exposure heterogeneity
    for v in ["low_dil", "rw_recreation_use", "rw_drinking_use", "rw_pathogen_impaired", "low_income_hi"]:
        sv = se[se[v].notna()]
        fit(f"het_{v}", g, sv, f"r1 + r2 + r1:{v} + r2:{v} + {TB} {FE}", "exceed")
    log(f"done {g}")

if MODE in ALL_GROUPS:
    pd.concat(rows, ignore_index=True).to_csv(OUT / f"model_results_{MODE}.csv", index=False)
    log(f"wrote model_results_{MODE}.csv")
    sys.exit(0)
res = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("model_results_*.csv"))], ignore_index=True)
res.to_csv(OUT / "model_results.csv", index=False)
log(f"combined {len(res)} rows")

# 4. rain-attributable violations by exposure category (main LPM, CSO-specific effects)
att = []
for g in ["ecoli", "fecal", "tss", "bod"]:
    r = res[(res["model"] == "cso") & (res["group"] == g) & (res["outcome"] == "exceed") & (res["how"] == "ols")]
    if r.empty:
        continue
    b = r.set_index("term")["Estimate"]
    s = p[(p["group"] == g) & p["exceed"].notna()].copy()
    s["att"] = (s["r1"] * (b.get("r1", 0) + s["cso"] * b.get("r1:cso", 0))
                + s["r2"] * (b.get("r2", 0) + s["cso"] * b.get("r2:cso", 0))).clip(lower=0)
    years = s["year"].nunique()
    for cat, mask in {
        "all plants": s.index == s.index,
        "combined-sewer city": s["cso"] == 1,
        "low dilution (DF<10)": s["low_dil"] == 1,
        "recreation-designated water": s["rw_recreation_use"] == 1,
        "drinking-water-designated water": s["rw_drinking_use"] == 1,
        "pathogen-impaired water": s["rw_pathogen_impaired"] == 1,
        "high low-income share (top third)": s["low_income_hi"] == 1,
    }.items():
        ss = s[mask]
        att.append(dict(group=g, category=cat, plants=ss["permit"].nunique(),
                        violation_months_per_year=ss["exceed"].sum() / years,
                        rain_attributable_per_year=ss["att"].sum() / years,
                        share_attributable=ss["att"].sum() / max(ss["exceed"].sum(), 1)))
pd.DataFrame(att).to_csv(OUT / "attributable.csv", index=False)
log("wrote attributable.csv")
