"""England: final-effluent samples at water-company sewage works (EA Water Quality Archive,
new API). Sampling points of type SA (SEWAGE DISCHARGES - FINAL/TREATED EFFLUENT - WATER
COMPANY); determinands BOD 5-day ATU (0085), ammoniacal nitrogen as N (0111), suspended solids
at 105 C (0135); 2016-2025. Output: raw/ea_points.csv, raw/ea_obs/<point>.csv.gz
"""
import gzip
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

B = "https://environment.data.gov.uk/water-quality"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "global/raw")
OBS = OUT / "ea_obs"
OBS.mkdir(parents=True, exist_ok=True)
DET = "0085,0111,0135"
UA = {"User-Agent": "Mozilla/5.0 (academic research script)"}


def get(url, accept, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept": accept})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            err = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"{url}: {err}")


def list_points():
    f = OUT / "ea_points.csv"
    if f.exists():
        return pd.read_csv(f)
    rows, skip = [], 0
    while True:
        d = json.loads(get(f"{B}/sampling-point?samplingPointType=SA&limit=250&skip={skip}", "application/ld+json"))
        for m in d["member"]:
            wkt = m.get("geometry", {}).get("asWKT", "")
            try:
                e, n = wkt.split("POINT(")[1].split(")")[0].split()
            except Exception:
                e = n = None
            rows.append(dict(point=m["notation"], name=m.get("prefLabel"), easting=e, northing=n,
                             status=(m.get("samplingPointStatus") or {}).get("prefLabel"),
                             region=(m.get("region") or {}).get("prefLabel")))
        skip += 250
        if skip >= d["totalItems"]:
            break
    p = pd.DataFrame(rows)
    p.to_csv(f, index=False)
    return p


def one(pt):
    try:
        return _one(pt)
    except Exception as e:  # log and move on; a rerun retries missing points
        print("FAILED", pt, e, flush=True)
        return 0


def _one(pt):
    f = OBS / f"{pt.replace('/', '_')}.csv.gz"
    if f.exists():
        return 0
    parts, skip = [], 0
    while True:
        txt = get(f"{B}/sampling-point/{urllib.parse.quote(pt)}/observation?dateFrom=2016-01-01&dateTo=2025-12-31"
                  f"&determinand={DET}&limit=250&skip={skip}", "text/csv").decode("utf-8", "replace")
        df = pd.read_csv(io.StringIO(txt), dtype=str) if txt.strip() else pd.DataFrame()
        if df.empty:
            break
        parts.append(df)
        if len(df) < 250:
            break
        skip += 250
    out = pd.concat(parts) if parts else pd.DataFrame(columns=["samplingPoint.notation"])
    keep = [c for c in ["samplingPoint.notation", "phenomenonTime", "samplingPurpose", "determinand.notation",
                        "result", "unit"] if c in out.columns]
    with gzip.open(f, "wt") as fh:
        out[keep].to_csv(fh, index=False)
    return len(out)


if __name__ == "__main__":
    pts = list_points()
    print("points:", len(pts), flush=True)
    done = 0
    with ThreadPoolExecutor(12) as ex:
        for i, n in enumerate(ex.map(one, pts["point"]), 1):
            done += n
            if i % 250 == 0:
                print(f"{i:,}/{len(pts):,} points, {done:,} rows", flush=True)
    print("finished", done, flush=True)
