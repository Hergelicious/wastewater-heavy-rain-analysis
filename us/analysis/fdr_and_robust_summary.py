"""Benjamini-Hochberg false-discovery-rate correction by pre-defined family of tests, and a
compact robustness summary. -> upgrades/fdr.csv, upgrades/robust_summary.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
U = W / "upgrades"


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


res = pd.read_csv(U / "model_results.csv")
res = res[res["how"] == "ols"]
ev = pd.read_csv(U / "cso_event.csv")
fam = []
sel = lambda m, terms, outc=("exceed",): res[(res["model"] == m) & res["term"].isin(terms) & res["outcome"].isin(outc)]
fam.append(("1 primary: heavy rain -> violations", sel("main", ["r1", "r2"])))
fam.append(("2 combined-sewer amplification", sel("cso", ["r1:cso", "r2:cso"])))
fam.append(("3 temperature (seasonal)", res[(res["model"] == "seasonal_temp") & res["term"].isin(["winter_t", "summer_t"])]))
pol = pd.concat([sel("cso_active_outfalls", ["r1:log_active", "r2:log_active"]),
                 sel("cso_share_closed", ["r1:closed_share", "r2:closed_share"]),
                 sel("cso_decree", ["r1:decree", "r2:decree"]),
                 sel("cso_decree_trend", ["r1:trend:decree", "r2:trend:decree"])])
e = ev[(ev["spec"] == "static") & (ev["term"] == "h_post")].assign(model="cso_event_study")
pol = pd.concat([pol, e.rename(columns={})])
fam.append(("4 CSO policy (exploratory)", pol))
het = res[res["model"].str.startswith("het_") & res["term"].str.contains(":")]
fam.append(("5 exposure heterogeneity", het))
rows = []
for name, d in fam:
    d = d.copy()
    d["family"] = name
    d["q_within_family"] = bh(d["Pr(>|t|)"])
    rows.append(d)
out = pd.concat(rows, ignore_index=True)
out["q_all_tests"] = bh(out["Pr(>|t|)"])
keep = ["family", "model", "group", "outcome", "term", "Estimate", "Std. Error", "Pr(>|t|)", "q_within_family", "q_all_tests"]
out[keep].to_csv(U / "fdr.csv", index=False)
pd.set_option("display.width", 220)
print(out.groupby("family").apply(lambda d: pd.Series({
    "tests": len(d), "p<0.05": int((d["Pr(>|t|)"] < 0.05).sum()), "q<0.05 (family)": int((d["q_within_family"] < 0.05).sum()),
    "q<0.05 (all)": int((d["q_all_tests"] < 0.05).sum())})).to_string())
sig = out[out["q_within_family"] < 0.1].sort_values(["family", "q_within_family"])
print(sig[keep].round(4).to_string(index=False))

# robustness summary (exceedance; pp)
rb = pd.concat([pd.read_csv(f) for f in sorted(U.glob("robust_*.csv")) if "summary" not in f.name])
rb["pp"] = rb["Estimate"] * 100
rb["lo"], rb["hi"] = rb["2.5%"] * 100, rb["97.5%"] * 100
rs = rb[rb["outcome"].isin(["exceed", "nonreceipt", "nodi_other"])][
    ["group", "check", "outcome", "term", "pp", "lo", "hi", "Pr(>|t|)", "plants", "base"]]
rs.to_csv(U / "robust_summary.csv", index=False)
print(rs[rs["group"].isin(["ecoli", "tss"])].round(3).to_string(index=False))
