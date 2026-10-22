"""Event study: does a plant's sensitivity to heavy rain change after a CSO storage/conveyance
project that sends captured combined sewage to it starts operating?

Treated plants: projects in analysis/cso_projects.csv that dewater to a named plant, dated to the
month, in service 2016-01..2024-12 (first such project per plant). Comparison: all other plants
in combined-sewer cities. Outcomes: pathogen (E. coli, else fecal coliform), TSS, BOD
exceedance; any of the three; and plant-standardised log peak concentration.
y ~ h + h:treated + post + h:post | plant x calendar month + state x year-month, h = heavy-rain
days (0-3, capped). SEs clustered by plant. Dynamic version: h x event-time bins.
-> upgrades/cso_event.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
HERE = Path(__file__).resolve().parent
OUT = W / "upgrades"

prj = pd.read_csv(HERE / "cso_projects.csv", dtype=str)
prj = prj[prj["dewaters_to_plant"].isin(["yes", "partly"])
          & prj["project_type"].str.contains("storage|tunnel|basin|reservoir|retention|conveyance")
          & prj["in_service_date"].str.match(r"^\d{4}-\d{2}$")
          & ~prj["plant_receiving_flow"].str.contains("unknown")]
prj["permit"] = prj["permit_ids"].str.split(";").str[0]
prj["d"] = pd.PeriodIndex(prj["in_service_date"], freq="M")
prj = prj[(prj["d"] >= pd.Period("2016-01", "M")) & (prj["d"] <= pd.Period("2024-12", "M"))]
treat = prj.groupby("permit")["d"].min()
print("treated plants:", len(treat))
print(treat.astype(str).to_string())

attr = pd.read_csv(W / "plants_attr.csv", dtype={"permit": str}, usecols=["permit", "cso_city"])
sample = set(attr.loc[attr["cso_city"] == 1, "permit"]) | set(treat.index)
cols = ["permit", "group", "ym", "cal_m", "exceed", "logc_max", "n_heavy", "STATE_CODE"]
p = pd.concat([c[c["permit"].isin(sample) & c["group"].isin(["ecoli", "fecal", "tss", "bod"])]
               for c in pd.read_csv(W / "panel.csv.gz", usecols=cols, chunksize=1_000_000,
                                    dtype={"permit": str, "ym": str, "STATE_CODE": str})])
# pathogen = E. coli where the plant reports it, otherwise fecal coliform
n = p[p["group"].isin(["ecoli", "fecal"])].groupby(["permit", "group"]).size().unstack(fill_value=0)
use_fecal = set(n.index[n.get("fecal", 0) > n.get("ecoli", 0)])
p = p[~((p["group"] == "ecoli") & p["permit"].isin(use_fecal)) & ~((p["group"] == "fecal") & ~p["permit"].isin(use_fecal))]
p["group"] = p["group"].replace({"ecoli": "pathogen", "fecal": "pathogen"})
# plant-standardised peak concentration
g = p.groupby(["permit", "group"])["logc_max"]
p["zmax"] = (p["logc_max"] - g.transform("mean")) / g.transform("std")
anyv = p.groupby(["permit", "ym"]).agg(exceed=("exceed", "max"), n_heavy=("n_heavy", "first"),
                                       cal_m=("cal_m", "first"), STATE_CODE=("STATE_CODE", "first")).reset_index()
anyv["group"] = "any (pathogen, TSS, BOD)"
p = pd.concat([p, anyv], ignore_index=True)

p["h"] = p["n_heavy"].clip(upper=3).astype(float)
p["treated"] = p["permit"].isin(treat.index).astype(float)
ymp = pd.PeriodIndex(p["ym"], freq="M")
td = p["permit"].map(treat)
et = pd.Series(np.nan, index=p.index)
m = td.notna()
et[m] = [(a - b).n for a, b in zip(ymp[m.to_numpy()], td[m])]
p["post"] = (et >= 0).astype(float)
for name, lo, hi in [("pre_far", -999, -13), ("e0", 0, 11), ("e1", 12, 23), ("e2", 24, 999)]:
    p[name] = ((et >= lo) & (et <= hi)).astype(float)
for v in ["treated", "post", "pre_far", "e0", "e1", "e2"]:
    p[f"h_{v}"] = p["h"] * p[v]

rows = []
for grp in ["pathogen", "tss", "bod", "any (pathogen, TSS, BOD)"]:
    for y in ["exceed", "zmax"]:
        d = p[(p["group"] == grp) & p[y].notna()].copy()
        if d.empty:
            continue
        d["pm"] = (d["permit"] + "_" + d["cal_m"].astype(str)).astype("category")
        d["sym"] = (d["STATE_CODE"] + "_" + d["ym"]).astype("category")
        for spec, f in [("static", "h + h_treated + post + h_post"),
                        ("dynamic", "h + h_treated + pre_far + e0 + e1 + e2 + h_pre_far + h_e0 + h_e1 + h_e2")]:
            try:
                mdl = pf.feols(f"{y} ~ {f} | pm + sym", data=d, vcov={"CRV1": "permit"})
                t = mdl.tidy().reset_index().rename(columns={"Coefficient": "term"})
                rows.append(t.assign(group=grp, outcome=y, spec=spec, n=mdl._N, plants=d["permit"].nunique(),
                                     treated_plants=int(d.loc[d["treated"] == 1, "permit"].nunique()),
                                     base=d[y].mean()))
            except Exception as e:
                print("FAILED", grp, y, spec, e)
res = pd.concat(rows, ignore_index=True)
res.to_csv(OUT / "cso_event.csv", index=False)
show = res[res["term"].str.startswith("h")]
pd.set_option("display.width", 220)
print(show[["group", "outcome", "spec", "term", "Estimate", "Std. Error", "Pr(>|t|)", "plants", "treated_plants"]]
      .round(4).to_string(index=False))
