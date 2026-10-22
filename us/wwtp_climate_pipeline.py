#!/usr/bin/env python3
"""
US data pipeline: municipal treatment plants, effluent monitoring and daily weather
==================================================================================

Builds the plant x month x parameter panel used in all US analyses.

Data
  * EPA ECHO ICIS-NPDES tables (plant list, coordinates, design flow, POTW flag)
  * EPA ECHO Discharge Monitoring Report (DMR) files, one per fiscal year
  * Daymet v4 daily weather (1 km), one request per plant (ORNL single-pixel API)

Steps (run all, or pick with --steps)
  download  ICIS-NPDES tables + DMR yearly zips          (~0.5 GB per year)
  plants    POTWs with coordinates
  dmr       stream the DMR zips, keep effluent-gross values for the analysed parameters
  climate   Daymet daily -> monthly climate, anomalies, heavy-rain-day counts
  panel     plant x month panel (concentrations, violations, flow, climate)

Usage
  python us/wwtp_climate_pipeline.py --state ALL --fy 2016-2025 --workdir wwtp_national
  python us/wwtp_climate_pipeline.py --state OH --workdir test_oh      # one-state test run

Needs pandas and numpy; downloads use the standard library. Every step caches its output in
--workdir, so a rerun skips finished work.
"""

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ECHO = "https://echo.epa.gov/files/echodownloads"
ICIS_ZIP = "npdes_downloads.zip"
DAYMET = "https://daymet.ornl.gov/single-pixel/api/data"
UA = {"User-Agent": "Mozilla/5.0 (academic research script)"}

# Parameter groups, matched on PARAMETER_DESC (lower case).
# kind C = concentration value, Q = quantity value (flow).
GROUPS = {
    "ammonia": (r"^nitrogen,\s*ammonia\s+total\s*\[as n\]", "C", "Ammonia-N"),
    "tp": (r"^phosphorus,\s*total(\s*\[as p\]|\s+as p)?\s*$", "C", "Total P"),
    "tn": (r"^nitrogen,\s*total(\s*\[as n\])?\s*$", "C", "Total N"),
    "tss": (r"^solids,\s*total suspended\s*$", "C", "TSS"),
    "bod": (r"^bod,\s*(5-day|carbonaceous\s*\[5 day)(?!.*percent)", "C", "BOD/CBOD"),
    "ecoli": (r"^e\.?\s*coli", "C", "E. coli"),
    "fecal": (r"^(coliform,\s*fecal|fecal coliform)(?!.*(%|percent|sludge))", "C", "Fecal coliform"),
    "flow": (r"^flow,\s*in conduit or thru treatment plant", "Q", "Effluent flow"),
}
POLLUTANTS = ["ammonia", "tp", "tn", "tss", "bod", "ecoli", "fecal"]
MAIN4 = ["ammonia", "tp", "tss", "ecoli"]  # main-figure panels (fallbacks below)

DMR_KEEP = [
    "EXTERNAL_PERMIT_NMBR", "PERM_FEATURE_NMBR", "PARAMETER_CODE", "PARAMETER_DESC",
    "MONITORING_LOCATION_CODE", "STATISTICAL_BASE_CODE", "STATISTICAL_BASE_TYPE_CODE",
    "VALUE_TYPE_CODE", "MONITORING_PERIOD_END_DATE", "NMBR_OF_REPORT", "DMR_VALUE_ID",
    "DMR_VALUE_QUALIFIER_CODE", "DMR_VALUE_STANDARD_UNITS", "STANDARD_UNIT_DESC",
    "LIMIT_VALUE_STANDARD_UNITS", "LIMIT_TYPE_CODE", "OPTIONAL_MONITORING_FLAG",
    "NODI_CODE", "EXCEEDANCE_PCT", "EXCEEDENCE_PCT", "VIOLATION_CODE",
]

# temperature-anomaly bins (deg C, monthly mean vs plant x calendar-month normal)
T_EDGES = [-np.inf, -3.0, -1.5, 1.5, 3.0, np.inf]
T_LABELS = ["< -3", "-3 to -1.5", "-1.5 to 1.5", "1.5 to 3", "> 3"]
T_REF = "-1.5 to 1.5"
R_LABELS = ["0", "1", "2+"]  # heavy-rain days (> plant's 99th pct daily rain) in month
R_REF = "0"
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------- download
def fetch(url, dest, tries=4):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  have {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(tries):
        try:
            log(f"  downloading {url}")
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got, step = 0, 0
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
                    if total and got * 10 // total > step:
                        step = got * 10 // total
                        log(f"    {got / 1e6:,.0f} / {total / 1e6:,.0f} MB")
            tmp.rename(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log(f"    attempt {k + 1} failed: {e}")
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"could not download {url}")


def step_download(a):
    raw = a.workdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    fetch(f"{ECHO}/{ICIS_ZIP}", raw / ICIS_ZIP)
    for fy in a.years:
        fetch(f"{ECHO}/npdes_dmrs_fy{fy}.zip", raw / f"npdes_dmrs_fy{fy}.zip")
    fetch(f"{ECHO}/ALL_CSO_downloads.zip", raw / "ALL_CSO_downloads.zip")
    # used by the plant-attribute, exposure and mechanism analyses
    for f in ["case_downloads.zip", "echo_demographics.zip", "npdes_attains_downloads.zip"]:
        fetch(f"{ECHO}/{f}", raw / f)
    fetch(f"{ECHO}/current_sewer_overflow_and_collection_systems_tables.zip", raw / "sso_current.zip")


def read_from_zip(zpath, name_pattern, **kw):
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if re.search(name_pattern, n, re.I)]
        if not names:
            raise FileNotFoundError(f"{name_pattern} not in {zpath.name}: {z.namelist()}")
        with z.open(names[0]) as fh:
            return pd.read_csv(fh, dtype=str, encoding="latin-1", **kw)


# ----------------------------------------------------------------------- plants
def step_plants(a):
    z = a.workdir / "raw" / ICIS_ZIP
    fac = read_from_zip(z, r"ICIS_FACILITIES\.csv$")
    per = read_from_zip(z, r"ICIS_PERMITS\.csv$")
    fac.columns, per.columns = fac.columns.str.upper(), per.columns.str.upper()

    per = per[per["VERSION_NMBR"].fillna("0").astype(float) == 0]
    per = per[per["FACILITY_TYPE_INDICATOR"].str.upper().str.strip() == "POTW"]
    per = per[["EXTERNAL_PERMIT_NMBR", "PERMIT_TYPE_CODE", "MAJOR_MINOR_STATUS_FLAG",
               "PERMIT_STATUS_CODE", "TOTAL_DESIGN_FLOW_NMBR", "ACTUAL_AVERAGE_FLOW_NMBR",
               "PERMIT_NAME"]].drop_duplicates("EXTERNAL_PERMIT_NMBR")

    fac = fac[["NPDES_ID", "FACILITY_NAME", "CITY", "STATE_CODE",
               "GEOCODE_LATITUDE", "GEOCODE_LONGITUDE"]].drop_duplicates("NPDES_ID")
    p = per.merge(fac, left_on="EXTERNAL_PERMIT_NMBR", right_on="NPDES_ID", how="left")
    p["lat"] = pd.to_numeric(p["GEOCODE_LATITUDE"], errors="coerce")
    p["lon"] = pd.to_numeric(p["GEOCODE_LONGITUDE"], errors="coerce")
    st = p["STATE_CODE"].fillna(p["EXTERNAL_PERMIT_NMBR"].str[:2]).str.upper()
    p["STATE_CODE"] = st
    if a.states != ["ALL"]:
        p = p[st.isin(a.states)]
    p = p[p["lat"].between(17, 72) & p["lon"].between(-180, -60)]
    p["design_mgd"] = pd.to_numeric(p["TOTAL_DESIGN_FLOW_NMBR"], errors="coerce")
    p["actual_mgd"] = pd.to_numeric(p["ACTUAL_AVERAGE_FLOW_NMBR"], errors="coerce")
    # placeholder or unit-error flows: exactly 999, or above the largest US plant (~1,500 MGD)
    for c in ["design_mgd", "actual_mgd"]:
        p.loc[(p[c] == 999) | (p[c] > 1500) | (p[c] <= 0), c] = np.nan
    p.loc[p["actual_mgd"] > 3 * p["design_mgd"], "actual_mgd"] = np.nan
    p["major"] = (p["MAJOR_MINOR_STATUS_FLAG"].str.upper() == "M").astype(int)
    p = p.rename(columns={"EXTERNAL_PERMIT_NMBR": "permit"})
    # combined sewer overflow (CSO) flags from EPA's national CSO inventory:
    # cso_permit = CSO outfalls are listed under this plant's own permit;
    # cso_city   = any CSO outfall is listed in the same city and state (separate permits)
    cz = a.workdir / "raw" / "ALL_CSO_downloads.zip"
    if cz.exists():
        cso = read_from_zip(cz, r"\.csv$")
        cso.columns = cso.columns.str.upper()
        p["cso_permit"] = p["permit"].isin(set(cso["NPDES_ID"])).astype(int)
        key = lambda c, s: c.fillna("").str.upper().str.strip() + "|" + s.fillna("").str.upper()
        cities = set(key(cso["FACILITY_CITY"], cso["STATE_CODE"]))
        p["cso_city"] = (key(p["CITY"], p["STATE_CODE"]).isin(cities) | (p["cso_permit"] == 1)).astype(int)
    else:
        p["cso_permit"] = p["cso_city"] = np.nan
    out = p[["permit", "PERMIT_NAME", "FACILITY_NAME", "CITY", "STATE_CODE", "lat", "lon",
             "design_mgd", "actual_mgd", "major", "cso_permit", "cso_city",
             "PERMIT_TYPE_CODE", "PERMIT_STATUS_CODE"]]
    out.to_csv(a.workdir / "plants.csv", index=False)
    log(f"plants: {len(out):,} POTW permits with coordinates in {','.join(a.states)} "
        f"({out['major'].sum():,} majors; {int(out['cso_permit'].sum()):,} with CSO outfalls "
        f"on their permit, {int(out['cso_city'].sum()):,} in a CSO city)")


# -------------------------------------------------------------------------- dmr
def classify(desc, cache={}):
    if desc not in cache:
        d = str(desc).lower().strip()
        cache[desc] = next((g for g, (rx, _, _) in GROUPS.items() if re.search(rx, d)), None)
    return cache[desc]


def dmr_files(a):
    """Per-state DMR extracts (or the single-file layout of earlier runs)."""
    files = sorted((a.workdir / "dmr").glob("dmr_*.csv.gz"))
    legacy = a.workdir / "dmr_extract.csv.gz"
    return files or ([legacy] if legacy.exists() else [])


def step_dmr(a):
    """One pass over each yearly national file; rows are appended to one file per
    state, so a national run never holds more than one chunk in memory."""
    import gzip
    plants = pd.read_csv(a.workdir / "plants.csv", dtype={"permit": str})
    state_of = dict(zip(plants["permit"], plants["STATE_CODE"].fillna("XX")))
    outdir = a.workdir / "dmr"
    outdir.mkdir(exist_ok=True)
    for f in outdir.glob("dmr_*.csv.gz"):
        f.unlink()
    handles, cols = {}, None
    desc_counts, loc_counts = pd.Series(dtype=float), pd.Series(dtype=float)
    total, permits_seen, groups = 0, set(), pd.Series(dtype=float)
    try:
        for fy in a.years:
            z = a.workdir / "raw" / f"npdes_dmrs_fy{fy}.zip"
            if not z.exists():
                log(f"  missing {z.name}; skipped")
                continue
            log(f"  scanning {z.name}")
            with zipfile.ZipFile(z) as zf:
                name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                with zf.open(name) as fh:
                    reader = pd.read_csv(fh, dtype=str, encoding="latin-1", chunksize=1_000_000,
                                         usecols=lambda c: c.strip().upper() in DMR_KEEP)
                    for ch in reader:
                        ch.columns = ch.columns.str.strip().str.upper()
                        if "EXCEEDENCE_PCT" in ch and "EXCEEDANCE_PCT" not in ch:
                            ch = ch.rename(columns={"EXCEEDENCE_PCT": "EXCEEDANCE_PCT"})
                        ch = ch[ch["EXTERNAL_PERMIT_NMBR"].isin(state_of)]
                        if ch.empty:
                            continue
                        desc_counts = desc_counts.add(ch["PARAMETER_DESC"].value_counts(), fill_value=0)
                        loc_counts = loc_counts.add(ch["MONITORING_LOCATION_CODE"].value_counts(), fill_value=0)
                        ch = ch[ch["MONITORING_LOCATION_CODE"].str.strip() == "1"]  # effluent gross
                        ch = ch.assign(group=ch["PARAMETER_DESC"].map(classify))
                        ch = ch[ch["group"].notna()].assign(fy=fy)
                        if "DMR_VALUE_ID" in ch:
                            ch = ch.drop_duplicates(["DMR_VALUE_ID", "VIOLATION_CODE"])
                        if cols is None:
                            cols = [c for c in DMR_KEEP if c in ch] + ["group", "fy"]
                            cols = [c if c != "EXCEEDENCE_PCT" else "EXCEEDANCE_PCT" for c in cols]
                            cols = list(dict.fromkeys(c for c in cols if c in ch))
                        ch = ch.reindex(columns=cols)
                        for st, sub in ch.groupby(ch["EXTERNAL_PERMIT_NMBR"].map(state_of)):
                            if st not in handles:
                                handles[st] = gzip.open(outdir / f"dmr_{st}.csv.gz", "wt")
                                handles[st].write(",".join(cols) + "\n")
                            sub.to_csv(handles[st], header=False, index=False)
                        total += len(ch)
                        permits_seen.update(ch["EXTERNAL_PERMIT_NMBR"].unique())
                        groups = groups.add(ch["group"].value_counts(), fill_value=0)
            log(f"    kept {total:,} rows so far")
    finally:
        for h in handles.values():
            h.close()
    if not total:
        sys.exit("no DMR rows found - check that DMR zips are in workdir/raw")
    (a.workdir / "dmr_permits.txt").write_text("\n".join(sorted(permits_seen)))
    desc_counts.sort_values(ascending=False).rename("rows").to_csv(a.workdir / "check_parameter_descriptions.csv")
    loc_counts.sort_values(ascending=False).rename("rows").to_csv(a.workdir / "check_monitoring_locations.csv")
    log(f"dmr: {total:,} effluent rows, {len(permits_seen):,} permits, {len(handles)} states")
    log("  group counts: " + ", ".join(f"{g}={int(n):,}" for g, n in groups.sort_values(ascending=False).items()))


# ---------------------------------------------------------------------- climate
def _land_candidates(lat, lon):
    """The plant's own point first, then nearby points: outfalls on the coast can fall on a
    Daymet water cell, which returns no data."""
    yield lat, lon
    for r in (0.02, 0.05, 0.1):
        for dy, dx in [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]:
            yield lat + dy * r, lon + dx * r


def daymet_one(lat, lon, y0, y1, dest):
    if dest.exists():
        return pd.read_csv(dest)
    tries = [(la, lo, y1) for la, lo in _land_candidates(lat, lon)]
    tries += [(lat, lon, end) for end in range(y1 - 1, y0, -1)]  # Daymet may lag ~1 year
    for la, lo, end in tries:
        url = (f"{DAYMET}?lat={la:.5f}&lon={lo:.5f}&vars=tmax,tmin,prcp"
               f"&start={y0}-01-01&end={end}-12-31")
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                continue
            raise
        if "year" not in txt.lower() or "nodata" in txt.lower()[:2000]:
            continue
        lines = txt.splitlines()
        start = next(i for i, s in enumerate(lines) if s.lower().startswith("year"))
        df = pd.read_csv(pd.io.common.StringIO("\n".join(lines[start:])))
        df.columns = [c.split(" ")[0].lower() for c in df.columns]
        df = df.rename(columns={"yday": "yday"})[["year", "yday", "tmax", "tmin", "prcp"]]
        df.to_csv(dest, index=False, compression="gzip")
        return df
    raise RuntimeError(f"Daymet returned nothing for {lat},{lon}")


def monthly_climate(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01") + pd.to_timedelta(df["yday"] - 1, "D")
    df["tmean"] = (df["tmax"] + df["tmin"]) / 2
    df["m"] = df["date"].dt.month
    p99 = df.loc[df["prcp"] > 0.1, "prcp"].quantile(0.99)
    hot = df.groupby("m")["tmax"].transform(lambda s: s.quantile(0.95))
    cold = df.groupby("m")["tmin"].transform(lambda s: s.quantile(0.05))
    df["heavy"], df["hot"], df["cold"] = df["prcp"] > p99, df["tmax"] > hot, df["tmin"] < cold
    df["ym"] = df["date"].dt.to_period("M")
    g = df.groupby("ym")
    return pd.DataFrame({
        "tmean": g["tmean"].mean(), "tmin_abs": g["tmin"].min(), "tmax_abs": g["tmax"].max(),
        "prcp_total": g["prcp"].sum(), "prcp_max1d": g["prcp"].max(),
        "n_heavy": g["heavy"].sum(), "n_hot": g["hot"].sum(), "n_cold": g["cold"].sum(),
    }).reset_index()


def step_climate(a):
    from concurrent.futures import ThreadPoolExecutor
    plants = pd.read_csv(a.workdir / "plants.csv", dtype={"permit": str})
    pf = a.workdir / "dmr_permits.txt"
    if pf.exists():
        keep = set(pf.read_text().split())
    else:
        keep = set(pd.read_csv(dmr_files(a)[0], usecols=["EXTERNAL_PERMIT_NMBR"], dtype=str)["EXTERNAL_PERMIT_NMBR"])
    plants = plants[plants["permit"].isin(keep)]
    cache = a.workdir / "daymet"
    cache.mkdir(exist_ok=True)
    y0, y1 = min(a.years) - 2, max(a.years)  # FY starts in October of the previous year
    log(f"  {len(plants):,} plants; {sum((cache / f'{p}.csv.gz').exists() for p in plants['permit']):,} cached")

    def one(r):
        for k in range(3):
            try:
                daily = daymet_one(r.lat, r.lon, y0, y1, cache / f"{r.permit}.csv.gz")
                m = monthly_climate(daily)
                m["permit"] = r.permit
                time.sleep(a.pause)
                return m
            except Exception as e:  # retry, then report and keep going
                err = e
                time.sleep(5 * (k + 1))
        log(f"  climate failed for {r.permit}: {err}")
        return None

    out = []
    with ThreadPoolExecutor(a.threads) as ex:
        for i, m in enumerate(ex.map(one, plants.itertuples(index=False)), 1):
            if m is not None:
                out.append(m)
            if i % 500 == 0:
                log(f"  climate {i:,}/{len(plants):,}")
    fails = len(plants) - len(out)
    clim = pd.concat(out, ignore_index=True)
    clim["cal_m"] = clim["ym"].dt.month
    norm = clim.groupby(["permit", "cal_m"])[["tmean", "prcp_total"]].transform("mean")
    clim["t_anom"] = clim["tmean"] - norm["tmean"]
    clim["p_anom_log"] = np.log((clim["prcp_total"] + 5) / (norm["prcp_total"] + 5))
    clim.to_csv(a.workdir / "climate_monthly.csv.gz", index=False)
    log(f"climate: {clim['permit'].nunique():,} plants, {len(clim):,} plant-months, {fails} failures")


# ------------------------------------------------------------------------ panel
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def parse_dates(s):
    d = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    miss = d.isna() & s.notna()
    if miss.any():
        d[miss] = pd.to_datetime(s[miss], errors="coerce")
    return d


def _series(c):
    """One consistent series per plant x group (dominant outfall, parameter code, statistic
    and unit); non-detects set to half the reported value or, for NODI "B" with no number,
    half the lowest detected value in that plant's series (a proxy for its detection limit)."""
    key = ["PERM_FEATURE_NMBR", "PARAMETER_CODE", "STATISTICAL_BASE_CODE", "STANDARD_UNIT_DESC"]
    c[key] = c[key].fillna("")
    n = c.groupby(["permit", "group"] + key).size().rename("n").reset_index()
    best = n.sort_values("n").groupby(["permit", "group"]).tail(1).drop(columns="n")
    c = c.merge(best, on=["permit", "group"] + key)
    lowest = c[~c["nd"] & (c["val"] > 0)].groupby(["permit", "group"])["val"].min().rename("dl")
    c = c.merge(lowest, on=["permit", "group"], how="left")
    c["val"] = np.where(c["nd"], c["val"].fillna(c["dl"]) / 2, c["val"])
    c = c[c["val"].notna()].drop(columns="dl")
    stats = c.groupby(["group", "STANDARD_UNIT_DESC"]).agg(n=("nd", "size"), nd=("nd", "sum")).reset_index()
    c = c.groupby(["permit", "group", "ym"]).agg(val=("val", "mean"), nd=("nd", "max")).reset_index()
    c = c[c["val"] >= 0]
    nmon = c.groupby(["permit", "group"])["ym"].transform("nunique")
    return c[nmon >= 24], stats  # at least two years of a series


def _panel_part(d):
    """Concentration, exceedance and flow series for one file of DMR rows."""
    d = d.assign(group=d["PARAMETER_DESC"].map(classify))  # re-apply current patterns
    d = d[d["group"].notna()]
    d["ym"] = parse_dates(d["MONITORING_PERIOD_END_DATE"]).dt.to_period("M")
    d = d[d["ym"].notna()]
    d["val"], d["lim"] = _num(d["DMR_VALUE_STANDARD_UNITS"]), _num(d["LIMIT_VALUE_STANDARD_UNITS"])
    d["exc_pct"] = _num(d.get("EXCEEDANCE_PCT", pd.Series(index=d.index, dtype=str)))
    code = d["NODI_CODE"].fillna("").str.strip().str.upper()
    below = code == "B"  # NODI "B" = below detection limit; some states (e.g. Ohio) use it
    nodi = (code != "") & ~below  # other NODI codes (no discharge, not required, ...) = missing
    d["nd"] = below | d["DMR_VALUE_QUALIFIER_CODE"].fillna("").str.contains("<")
    d["enf"] = (d["lim"].notna() & (d["LIMIT_TYPE_CODE"].fillna("ENF").str.upper() == "ENF")
                & (d["OPTIONAL_MONITORING_FLAG"].fillna("N").str.upper() != "Y"))
    d["exceed"] = d["enf"] & ((d["VIOLATION_CODE"].fillna("") == "E90") | (d["exc_pct"] > 0))
    d = d.rename(columns={"EXTERNAL_PERMIT_NMBR": "permit"})

    # exceedance panel: any enforceable limit exceeded in that plant-month (any statistic)
    # only months with a submitted value (or a below-detection code) count; missing reports
    # (non-receipt) are not treated as compliance
    e = d[d["enf"] & ~nodi & (d["val"].notna() | below)].groupby(["permit", "group", "ym"])["exceed"].max().astype(int)
    e = e.rename("exceed").reset_index()

    # concentration panels: one consistent series per plant x group, for the monthly
    # average (AVG) and for the highest daily value reported in the month (MAX = peak)
    kind = d["group"].map({g: v[1] for g, v in GROUPS.items()})
    base = d[~nodi & (d["val"].notna() | below)
             & (d["VALUE_TYPE_CODE"].fillna("").str.upper().str[:1] == kind.fillna("?"))
             & (_num(d["NMBR_OF_REPORT"]).fillna(1) == 1)]
    stype = base["STATISTICAL_BASE_TYPE_CODE"].fillna("").str.upper()
    c, stats = _series(base[stype == "AVG"].copy())
    cmax, _ = _series(base[stype == "MAX"].copy())
    c = c.merge(cmax.rename(columns={"val": "vmax", "nd": "nd_max"}), on=["permit", "group", "ym"],
                how="outer")

    # flow -> hydraulic loading relative to design flow
    flow = c[(c["group"] == "flow") & c["val"].notna()][["permit", "ym", "val"]].rename(columns={"val": "flow_mgd"})
    c = c[c["group"] != "flow"]
    return c, e, flow, stats


def step_panel(a):
    plants = pd.read_csv(a.workdir / "plants.csv", dtype={"permit": str})
    cs, es, fs, ss = [], [], [], []
    for f in dmr_files(a):
        d = pd.read_csv(f, dtype=str)
        c, e, flow, stats = _panel_part(d)
        cs.append(c), es.append(e), fs.append(flow), ss.append(stats)
        log(f"  {f.name}: {c['permit'].nunique():,} plants")
        del d
    c, e, flow, stats = (pd.concat(x, ignore_index=True) for x in (cs, es, fs, ss))
    stats = stats.groupby(["group", "STANDARD_UNIT_DESC"])[["n", "nd"]].sum().reset_index()
    units = stats.sort_values("n").groupby("group")["STANDARD_UNIT_DESC"].last()
    nd_share = stats.groupby("group")["nd"].sum() / stats.groupby("group")["n"].sum()


    clim = pd.read_csv(a.workdir / "climate_monthly.csv.gz", dtype={"permit": str})
    clim["ym"] = pd.PeriodIndex(clim["ym"], freq="M")
    panel = (c.merge(e, on=["permit", "group", "ym"], how="outer")
             .merge(clim, on=["permit", "ym"], how="inner")
             .merge(flow, on=["permit", "ym"], how="left")
             .merge(plants[["permit", "STATE_CODE", "design_mgd", "major", "cso_permit", "cso_city",
                            "lat", "lon"]], on="permit", how="left"))
    panel["hyd_ratio"] = panel["flow_mgd"] / panel["design_mgd"].where(panel["design_mgd"] > 0)
    floor = panel.groupby("group")["val"].transform(lambda s: s[s > 0].quantile(0.01))
    panel["logc"] = np.log10(panel["val"].clip(lower=floor))
    floor_max = panel.groupby("group")["vmax"].transform(lambda s: s[s > 0].quantile(0.01))
    panel["logc_max"] = np.log10(panel["vmax"].clip(lower=floor_max))
    panel["year"] = panel["ym"].dt.year
    panel["season"] = panel["cal_m"].map(SEASONS)
    panel["t_bin"] = pd.cut(panel["t_anom"], T_EDGES, labels=T_LABELS, right=False)
    panel["r_bin"] = pd.cut(panel["n_heavy"], [-0.5, 0.5, 1.5, np.inf], labels=R_LABELS)
    panel.to_csv(a.workdir / "panel.csv.gz", index=False)

    lines = ["# Feasibility summary", "",
             f"States: {', '.join(a.states)}; fiscal years {min(a.years)}-{max(a.years)}", "",
             "| group | plants (conc.) | plant-months (conc.) | unit | non-detect share | "
             "plants (limit) | plant-months (limit) | exceedance rate |",
             "|---|---|---|---|---|---|---|---|"]
    for g in POLLUTANTS:
        sc = panel[(panel["group"] == g) & panel["val"].notna()]
        se = panel[(panel["group"] == g) & panel["exceed"].notna()]
        if sc.empty and se.empty:
            continue
        lines.append(f"| {g} | {sc['permit'].nunique():,} | {len(sc):,} | {units.get(g, '')} | "
                     f"{nd_share.get(g, np.nan):.1%} | {se['permit'].nunique():,} | {len(se):,} | "
                     f"{se['exceed'].mean():.2%} |")
    lines += ["", "Temperature-anomaly bin counts (all groups pooled):", "",
              panel["t_bin"].value_counts().reindex(T_LABELS).to_string(), "",
              "Heavy-rain-day bin counts:", "", panel["r_bin"].value_counts().reindex(R_LABELS).to_string()]
    (a.workdir / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"panel: {len(panel):,} plant-month-group rows, {panel['permit'].nunique():,} plants")
    print("\n".join(lines))


# ------------------------------------------------------------------------- main
STEPS = ["download", "plants", "dmr", "climate", "panel"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="OH", help="state code(s), comma-separated (OH,IN,MI) or ALL")
    ap.add_argument("--fy", default="2016-2025", help="fiscal-year range, e.g. 2016-2025")
    ap.add_argument("--workdir", default="wwtp_climate", type=Path)
    ap.add_argument("--steps", default="all", help="comma-separated subset of: " + ",".join(STEPS))
    ap.add_argument("--min-plants", type=int, default=30, help="skip a parameter with fewer plants")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between Daymet requests")
    ap.add_argument("--threads", type=int, default=4, help="parallel Daymet requests")
    a = ap.parse_args(argv)
    a.states = [s.strip().upper() for s in a.state.split(",")]
    y0, y1 = (int(x) for x in a.fy.split("-"))
    a.years = list(range(y0, y1 + 1))
    a.workdir.mkdir(parents=True, exist_ok=True)
    steps = STEPS if a.steps == "all" else [s.strip() for s in a.steps.split(",")]
    for s in steps:
        log(f"== {s}")
        globals()[f"step_{s}"](a)
    log("done")


if __name__ == "__main__":
    main()
