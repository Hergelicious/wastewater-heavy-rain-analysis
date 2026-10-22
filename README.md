# Heavy rain and municipal wastewater treatment

## Layout

| Folder | Contents |
|---|---|
| `us/` | data pipeline (`wwtp_climate_pipeline.py`) and US analyses (`us/analysis/`) |
| `england/` | England replication: download, the test with criteria fixed in advance (`england_preregistration.md`), robustness and sampling checks |
| `global/` | HydroWASTE linkage, US effects by treatment level, CMIP6 heavy-rain days and the global projection |

## Requirements

Python 3.10 or newer with the packages in `requirements.txt` (`pip install -r requirements.txt` into an existing
environment works; no separate environment is required). The national run needs about 8 GB of RAM and about 10 GB of disk while the DMR files are processed.

## Data

All inputs are public. The pipeline downloads the EPA and Daymet data itself

| Source | Where it goes | Used for |
|---|---|---|
| EPA ECHO ICIS-NPDES, Discharge Monitoring Reports FY2016–2025, CSO inventory, enforcement cases, NPDES–ATTAINS, facility demographics, sewer overflow and bypass reports (echo.epa.gov/tools/data-downloads) | downloaded to `$US_DIR/raw/` | plants, effluent values, limits, violations, plant attributes, bypasses |
| Daymet v4 daily weather, 1 km (daymet.ornl.gov, single-pixel API) | downloaded to `$US_DIR/daymet/` | rain and temperature at each US plant |
| HydroRIVERS v1.0, North America (hydrosheds.org) | `HydroRIVERS_v10_na_shp/` in the working directory | dilution factors |
| Environment Agency Water Quality Archive (environment.data.gov.uk/water-quality) | downloaded by `england/england_download.py` to `$GLOBAL_DIR/raw/` | final-effluent BOD, ammonia and suspended solids |
| E-OBS v33.0e daily precipitation, 0.1° ensemble mean (Copernicus Climate Change Service / ECA&D) | `$GLOBAL_DIR/raw/rr_ens_mean_0.1deg_reg_2011-2025_v33.0e.nc` | rain at each English works |
| HydroWASTE v1.0 (hydrosheds.org) | `$GLOBAL_DIR/hydrowaste/HydroWASTE_v10.csv` | 58,502 treatment plants worldwide |
| NEX-GDDP-CMIP6 daily precipitation, 0.25° (registry.opendata.aws/nex-gddp-cmip6) | streamed by `global/cmip6_heavy_rain.py` | heavy-rain days, 15 models |
| UN M49 country codes (github.com/datasets/country-codes) and Natural Earth 1:50m admin-0 countries | `$GLOBAL_DIR/ext/country-codes.csv`, `$GLOBAL_DIR/ext/ne/` | regions, least developed countries, income groups |
| In-service dates of combined-sewer storage projects, each with its source | `us/analysis/cso_projects.csv` (compiled for this study) | storage before–after analysis |

Set `US_DIR` (default `wwtp_national`) and `GLOBAL_DIR` (default `global`) to the data folders.

## Run order

```bash
export US_DIR=wwtp_national GLOBAL_DIR=global

# 1. US panel: downloads, plants, DMR extraction, Daymet weather, plant x month panel
python us/wwtp_climate_pipeline.py --state ALL --fy 2016-2025 --workdir $US_DIR

# 2. plant attributes (combined sewers, decrees, receiving waters, demographics, dilution)
python us/analysis/dilution.py $US_DIR
python us/analysis/plant_attributes.py $US_DIR

# 3. main models, one process per parameter, then rain-attributable violations
bash us/analysis/run_upgrades.sh $US_DIR

# 4. robustness, storage projects, bypasses, multiple testing, sample counts
python us/analysis/robust_inputs.py $US_DIR
for g in ecoli fecal tss bod ammonia tp; do python us/analysis/robust_models.py $US_DIR $g; done
python us/analysis/cso_event.py $US_DIR
python us/analysis/bypass_mechanism.py $US_DIR
python us/analysis/fdr_and_robust_summary.py $US_DIR
python us/analysis/sample_counts.py $US_DIR
python us/analysis/dataflow.py $US_DIR

# 5. loads, flow and transferability; baseline utilisation; simple warming scaling
bash us/analysis/run_transfer_tests.sh $US_DIR
for g in tss ecoli fecal bod; do python us/analysis/capacity_test.py $US_DIR $g; done
python us/analysis/projection_shares.py $US_DIR
python us/analysis/projections.py $US_DIR

# 6. England replication
python england/england_download.py $GLOBAL_DIR/raw
python england/england_analysis.py
python england/england_robust.py
python england/england_sampling.py

# 7. global outlook
python global/link_hydrowaste.py
python global/transfer_effects.py
python global/transfer_vcov.py
python global/level_difference.py
python global/cmip6_heavy_rain.py          
python global/global_projection.py         
python global/global_stats.py
python global/us_consistency.py
python global/global_sensitivity.py
```

Every step caches its output, so a rerun skips finished work. `--state OH` gives a quick one-state test of the pipeline.



## Main specification

Plant-month outcomes: any exceedance of an enforceable permit limit for the parameter, at any outfall and for any
statistic (linear probability; Poisson for *E. coli*, TSS and ammonia); log10 monthly-average and monthly-peak
concentration from one dominant series per plant; log10 monthly-average daily load and flow. Exposure: number of days
above the plant's own 99th percentile of wet-day (>0.1 mm) rain, binned 0, 1, ≥2, with monthly temperature-anomaly bins.
Fixed effects: plant × calendar month and state × year-month; standard errors clustered by state.

## Licence

MIT (see `LICENSE`). Data remain under their providers' terms.
