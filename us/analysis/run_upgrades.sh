#!/bin/bash
# one process per pollutant keeps memory bounded; then combine and attribute
W=${1:-wwtp_national}
D=$(cd "$(dirname "$0")" && pwd)
for g in ecoli fecal tss bod ammonia tp; do
  [ -f "$W/upgrades/model_results_$g.csv" ] && { echo "have $g"; continue; }
  python3 -I "$D"/upgrade_models.py "$W" $g || echo "FAILED $g"
done
python3 -I "$D"/upgrade_models.py "$W" attributable
