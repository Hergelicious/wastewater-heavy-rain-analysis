"""England robustness: (1) exceedance of the Urban Waste Water Treatment Directive limits (BOD 25 mg/L,
suspended solids 35 mg/L) as a regulatory outcome; (2) the main models by sampling programme
(operator self-monitoring vs UWWTD monitoring vs other Environment Agency samples); (3) placebo for
the regulatory outcome. Same fixed effects as the main test (works x calendar month, region x ISO
week; SEs by works). Output: out/england_robust.csv
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
G = Path(os.environ.get("GLOBAL_DIR", "global"))
OUT = G / "out"
pan = pd.read_csv(OUT / "england_panel.csv.gz", usecols=["point", "group", "date", "val", "purpose", "h0_2",
                                                         "h_lead", "logc", "pm", "rw"])
pan = pan[pan["group"].isin(["bod", "tss"])].dropna(subset=["h0_2"])
LIMIT = {"bod": 25.0, "tss": 35.0}
pan["over_limit"] = (pan["val"] > pan["group"].map(LIMIT)).astype(float)
pan["programme"] = np.select(
    [pan["purpose"].str.contains("SELF MONITORING", na=False), pan["purpose"].str.contains("UWWTD", na=False)],
    ["operator self-monitoring", "UWWTD monitoring"], "other Environment Agency samples")
for c in ["pm", "rw"]:
    pan[c] = pan[c].astype("category")
rows = []


def fit(d, y, f, label):
    try:
        m = pf.feols(f"{y} ~ {f} | pm + rw", data=d, vcov={"CRV1": "point"})
    except ValueError as exc:  # too few repeat samples within works-month cells
        print("skipped", g, label, str(exc).strip()[:60], flush=True)
        return
    t = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    rows.append(t.assign(group=g, spec=label, outcome=y, n=m._N, works=d["point"].nunique(), base=d[y].mean()))


for g in ["tss", "bod"]:
    d = pan[pan["group"] == g]
    fit(d, "over_limit", "h0_2", "UWWTD limit exceedance")
    fit(d.dropna(subset=["h_lead"]), "over_limit", "h0_2 + h_lead", "UWWTD limit exceedance + placebo")
    for prog, dp in d.groupby("programme"):
        if dp["point"].nunique() >= 100:
            fit(dp, "logc", "h0_2", f"log concentration, {prog}")
    print("done", g, flush=True)
res = pd.concat(rows, ignore_index=True)
lin = res["outcome"] == "over_limit"
res["pct"] = np.where(lin, 100 * res["Estimate"], (10 ** res["Estimate"] - 1) * 100)
res["lo"] = np.where(lin, 100 * res["2.5%"], (10 ** res["2.5%"] - 1) * 100)
res["hi"] = np.where(lin, 100 * res["97.5%"], (10 ** res["97.5%"] - 1) * 100)
res["relative_pct"] = np.where(lin, 100 * res["Estimate"] / res["base"], np.nan)
res.to_csv(OUT / "england_robust.csv", index=False)
pd.set_option("display.width", 250)
print(res[["group", "spec", "term", "pct", "lo", "hi", "Pr(>|t|)", "relative_pct", "base", "n", "works"]]
      .round(4).to_string(index=False))
