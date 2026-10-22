# England replication test: criteria fixed before analysis (written 2026-10-21, before any England result was seen)

Data: EA Water Quality Archive, final-effluent samples at water-company sewage works (sampling point
type SA), 2016-2025; BOD 5-day ATU, ammoniacal nitrogen as N, suspended solids. Rain: E-OBS v33.0e
daily 0.1 deg, nearest cell to each works.

Exposure: any day in the sample day or the two days before it with rain above the works' own 99th
percentile of wet-day rain (2011-2025). Outcome: log10 concentration (non-detects at half the
reported limit). Fixed effects: works x calendar month, and EA region x calendar week (removes
regional shocks). Standard errors clustered by works.

GO if both:
1. Suspended solids or BOD rises after heavy rain: positive coefficient with P < 0.01.
2. Placebo (heavy rain in the three days after the sample) is not significantly positive (P > 0.05
   or negative), for the outcome(s) that pass criterion 1.

NO-GO otherwise. Ammonia is reported but not part of the decision (in the US it did not respond).
Secondary (not part of the decision): per-day dose response, an excursion outcome (sample above
the works' own 90th percentile), and works with a storm-tank sampling point vs without.
