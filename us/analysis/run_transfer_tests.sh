#!/bin/bash
# one process per pollutant keeps memory low
W=${1:-wwtp_national}
D=$(cd "$(dirname "$0")" && pwd)
for g in tss ecoli fecal bod; do python3 -I "$D"/transfer_tests.py $W $g > $W/upgrades/transfer_tests_$g.log 2>&1; done
echo all-done
