"""Inputs for robustness checks -> upgrades/alt_rain.csv.gz, upgrades/reporting.csv.gz

alt_rain:  per plant-month, days above the plant's 95th and 99.9th percentile wet-day rain,
           and days with >= 50 mm (from the same Daymet record as the main analysis)
reporting: per plant-pollutant-month, whether EPA recorded a missing report (non-receipt
           violation D80/D90) or a "no data" code other than below-detection (B) and no
           discharge (C)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

W = Path(sys.argv[1] if len(sys.argv) > 1 else "wwtp_national")
OUT = W / "upgrades"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rain_one(path):
    d = pd.read_csv(path)
    wet = d.loc[d["prcp"] > 0.1, "prcp"]
    q95, q999 = wet.quantile(0.95), wet.quantile(0.999)
    date = pd.to_datetime(d["year"].astype(str) + "-01-01") + pd.to_timedelta(d["yday"] - 1, "D")
    g = pd.DataFrame({"ym": date.dt.to_period("M").astype(str), "n95": d["prcp"] > q95,
                      "n999": d["prcp"] > q999, "n50": d["prcp"] >= 50}).groupby("ym").sum()
    g["permit"] = path.name.replace(".csv.gz", "")
    return g.reset_index()


def report_one(f):
    import wwtp_climate_pipeline as w
    d = pd.read_csv(f, dtype=str, usecols=["EXTERNAL_PERMIT_NMBR", "PARAMETER_DESC", "MONITORING_PERIOD_END_DATE",
                                          "NODI_CODE", "VIOLATION_CODE"])
    d["group"] = d["PARAMETER_DESC"].map(w.classify)
    d = d[d["group"].isin(["ecoli", "fecal", "tss", "bod", "ammonia", "tp"])]
    d["ym"] = w.parse_dates(d["MONITORING_PERIOD_END_DATE"]).dt.to_period("M").astype(str)
    code = d["NODI_CODE"].fillna("").str.strip().str.upper()
    d["nonreceipt"] = d["VIOLATION_CODE"].fillna("").isin(["D80", "D90"])
    d["nodi_other"] = (code != "") & ~code.isin(["B", "C"])
    return (d.groupby(["EXTERNAL_PERMIT_NMBR", "group", "ym"])[["nonreceipt", "nodi_other"]].max()
            .astype(int).reset_index().rename(columns={"EXTERNAL_PERMIT_NMBR": "permit"}))


if __name__ == "__main__":
    keep = set((W / "dmr_permits.txt").read_text().split())
    files = [f for f in (W / "daymet").glob("*.csv.gz") if f.name.replace(".csv.gz", "") in keep]
    with ProcessPoolExecutor(2) as ex:
        rain = pd.concat(ex.map(rain_one, files, chunksize=50), ignore_index=True)
    rain.to_csv(OUT / "alt_rain.csv.gz", index=False)
    print("alt_rain:", len(rain), rain[["n95", "n999", "n50"]].mean().round(3).to_dict(), flush=True)
    with ProcessPoolExecutor(2) as ex:
        rep = pd.concat(ex.map(report_one, sorted((W / "dmr").glob("dmr_*.csv.gz"))), ignore_index=True)
    rep.to_csv(OUT / "reporting.csv.gz", index=False)
    print("reporting:", len(rep), rep[["nonreceipt", "nodi_other"]].mean().round(4).to_dict())
