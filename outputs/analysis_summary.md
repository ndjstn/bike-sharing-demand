# Bike-sharing demand summary

Rows: 17,379. Date range: 2011-01-01 to 2012-12-31.

## CV strategy comparison (5 folds, RMSLE)

| Model | Random k-fold | Time-series CV |
|---|---:|---:|
| naive_seasonal | 0.6438 | 0.5086 |
| ridge | 1.1511 | 1.2300 |
| lightgbm | 0.4055 | 0.5946 |