# When a Naive Baseline Beats LightGBM: Bike-Sharing Demand with Proper Time-Series CV

UCI hourly bike-sharing data, two cross-validation strategies, three models. The single finding that's worth the project: random 5-fold makes LightGBM look like the winner at 0.405 RMSLE. Time-series 5-fold says LightGBM sits at 0.595 and a naive "mean `cnt` by (weekday, hour)" baseline beats it at 0.509.

## Key results

| Model | Random 5-fold RMSLE | Time-series 5-fold RMSLE |
| --- | ---: | ---: |
| Naive seasonal (weekday × hour) | 0.644 | **0.509** |
| Ridge with cyclical features | 1.151 | 1.230 |
| LightGBM with seasonal features | **0.405** | 0.595 |

Validation strategy matters more than model choice when your data has time structure. Random k-fold lets training folds contain information from the future relative to validation folds, which is leakage dressed up as generalisation.

## What is in this repo

`src/run_analysis.py` is the end-to-end pipeline — it fits the naive baseline, Ridge, and LightGBM under both random k-fold and TimeSeriesSplit and writes the CV comparison table. `scripts/build_teaching_animation.py` regenerates the two-panel teaching animation that contrasts expanding-window folds with random folds. `notebooks/analysis.ipynb` is the Kaggle-publishable narrative walk-through. `figures/` carries the grouped-bar CV-comparison hero, an hour-by-weekday heatmap, the monthly seasonality line, the temperature-effect curve, and the teaching animation. `outputs/` holds the CV comparison table.

`REPORT.md` is the long-form analysis.

## How to reproduce

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_analysis.py --data data/hour.csv --figures figures --outputs outputs
```

Dataset: `hour.csv` from the UCI Bike Sharing Dataset mirror on Kaggle ([Fanaee-T & Gama, 2014](https://doi.org/10.1007/s13748-013-0040-3)). Download from <https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset> and place at `data/hour.csv`.

## Further reading

<https://ndjstn.github.io/posts/bike-sharing-timeseries-cv/>.

## License

MIT.
