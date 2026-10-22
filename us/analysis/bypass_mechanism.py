"""Mechanism check with EPA's NPDES eRule sewer overflow and bypass reports (mostly 2025 on).

Treatment-plant bypass events per plant, by sewer type (separate / combined-sewer city / under a
federal CSO consent decree), and the share of bypasses that fall on or the day after a heavy-
rain day at the plant (Daymet, 2025). Descriptive: the reporting period is short and reporting
requirements differ between states. -> upgrades/bypass_summary.csv, upgrades/bypass_poisson.csv
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
OUT = W / "upgrades"
z = zipfile.ZipFile(W / "raw" / "sso_current.zip")
rd = lambda n: pd.read_csv(z.open(n), dtype=str, encoding="latin-1")
ev = rd("sewer_overflow_bypass_report_events.csv")
ty = rd("sewer_overflow_bypass_types.csv")[["sewer_overflow_bypass_event_key", "sewer_overflow_bypass_type_code"]]
ev = ev.merge(ty, on="sewer_overflow_bypass_event_key", how="left")
ev["start"] = pd.to_datetime(ev["sewer_overflow_bypass_start_datetime"].fillna(ev["program_report_start_date"]),
                             errors="coerce")
ev = ev[ev["start"].notna()]
first, last = ev["start"].min(), ev["start"].max()
months = max((last - first).days / 30.44, 1)

attr = pd.read_csv(W / "plants_attr.csv", dtype={"permit": str})
attr["type"] = np.select([attr["cso_decree"] == 1, attr["cso_city"] == 1], ["CSO city, federal decree", "CSO city, no decree"],
                         "separate sewers")
cs = rd("collection_system_permits.csv")
cs["pct_css"] = pd.to_numeric(cs["percent_collection_system_css"], errors="coerce")
attr = attr.merge(cs.groupby("permit_identifier")["pct_css"].max().rename("pct_css"), left_on="permit",
                  right_index=True, how="left")

byp = ev[ev["sewer_overflow_bypass_type_code"] == "BYP"]
cnt = byp.groupby("permit_identifier").size().rename("bypasses")
wet = byp[byp["wet_weather_occurance_indicator"].fillna("").str.upper().isin(["Y", "YES", "TRUE", "1"])]
attr = attr.merge(cnt, left_on="permit", right_index=True, how="left")
attr = attr.merge(wet.groupby("permit_identifier").size().rename("wet_bypasses"), left_on="permit",
                  right_index=True, how="left")
attr[["bypasses", "wet_bypasses"]] = attr[["bypasses", "wet_bypasses"]].fillna(0)
reporters = set(ev["permit_identifier"])  # plants that filed any eRule overflow/bypass report
attr["files_reports"] = attr["permit"].isin(reporters)

summ = attr.groupby("type").agg(plants=("permit", "size"), plants_filing=("files_reports", "sum"),
                                bypass_events=("bypasses", "sum"), wet_weather_bypasses=("wet_bypasses", "sum"))
summ["bypasses_per_100_plants_per_year"] = summ["bypass_events"] / summ["plants"] * 100 * 12 / months
summ["wet_share"] = summ["wet_weather_bypasses"] / summ["bypass_events"].where(summ["bypass_events"] > 0)
summ.to_csv(OUT / "bypass_summary.csv")
print(f"events {len(ev):,} ({first.date()} to {last.date()}), bypasses {len(byp):,}")
print(summ.round(3).to_string())

# Poisson across plants: bypass counts by sewer type, controlling for size and state
a = attr[attr["design_mgd"].notna()].copy()
a["log_design"] = np.log1p(a["design_mgd"])
a["cso_nodecree"] = ((a["cso_city"] == 1) & (a["cso_decree"] == 0)).astype(float)
a["decree"] = a["cso_decree"].fillna(0)
a = a[a.groupby("STATE_CODE")["bypasses"].transform("sum") > 0]
m = pf.fepois("bypasses ~ cso_nodecree + decree + log_design | STATE_CODE", data=a, vcov={"CRV1": "STATE_CODE"})
t = m.tidy().reset_index()
t["rate_ratio"] = np.exp(t["Estimate"])
t.to_csv(OUT / "bypass_poisson.csv", index=False)
print(t[["Coefficient", "Estimate", "Std. Error", "Pr(>|t|)", "rate_ratio"]].round(3).to_string(index=False))

# timing: share of bypasses on or the day after a heavy-rain day at the plant
rows = []
for r in byp.itertuples():
    f = W / "daymet" / f"{r.permit_identifier}.csv.gz"
    if not f.exists() or r.start.year > 2025:
        continue
    d = pd.read_csv(f)
    wetd = d.loc[d["prcp"] > 0.1, "prcp"]
    p99 = wetd.quantile(0.99)
    date = pd.to_datetime(d["year"].astype(str) + "-01-01") + pd.to_timedelta(d["yday"] - 1, "D")
    heavy = set(date[d["prcp"] > p99].dt.date)
    rainy = set(date[d["prcp"] > 0.1].dt.date)
    day = r.start.date()
    near = {day, (r.start - pd.Timedelta(days=1)).date()}
    rows.append(dict(permit=r.permit_identifier, heavy=bool(near & heavy), rainy=bool(near & rainy),
                     base_heavy=2 * (d["prcp"] > p99).mean()))
tm = pd.DataFrame(rows)
if len(tm):
    print(f"bypasses with Daymet coverage: {len(tm)}; on/after a heavy-rain day: {tm['heavy'].mean():.1%} "
          f"(expected by chance ~{tm['base_heavy'].mean():.1%}); on/after any rain: {tm['rainy'].mean():.1%}")
    tm.to_csv(OUT / "bypass_timing.csv", index=False)
