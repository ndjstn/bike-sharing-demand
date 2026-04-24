# When a Naive Baseline Beats LightGBM: Bike-Sharing Demand with Proper Time-Series Cross-Validation

Running the UCI hourly bike-sharing dataset through a random 5-fold cross-validation, LightGBM comes out at Log-RMSLE 0.405 and looks like the obvious winner. Running the same data through a `TimeSeriesSplit` — where each fold's training set is everything before a certain date and the validation set is what comes next — LightGBM lands at 0.595 and gets beaten by a naive "seasonal mean of (weekday, hour)" baseline at 0.509. That is a 46.6 percent jump in LightGBM's error when the validation strategy stops leaking the future into training.

Validation strategy matters more than model choice when the data has time structure. Random k-fold shuffles the time dimension away, which means the training fold always contains observations from the future relative to the validation fold. The model learns from information it will not have at inference time, and the cross-validation score flatters it accordingly.

## The dataset

17,379 hourly observations from the UCI Bike Sharing Dataset (Capital Bikeshare, Washington DC, 2011-2012), 14 weather and calendar features per observation. The target is `cnt`, total rentals in that hour. Data runs continuously from 2011-01-01 to 2012-12-31. Median hourly demand is 142 rentals, and the interquartile range runs from 40 to 281.

![Hour-weekday heatmap of mean bike rentals, with weekday commuter spikes and weekend afternoon peaks.](figures/hour-weekday-heatmap.png)

Weekday mornings hit a sharp 8am peak, a smaller noon bump, and a bigger 5-6pm peak — classic commuter pattern. Weekends are flatter with a single broad afternoon bulge from 11am to 6pm. That bimodal weekday pattern versus the unimodal weekend pattern is why a naive seasonal baseline indexed on (weekday, hour) does so well.

![Monthly rental volume across the two-year range showing strong seasonality and year-over-year growth.](figures/monthly-seasonality.png)

The monthly totals trace a seasonal shape — summer peaks, winter troughs — and grow about 65 percent between 2011 and 2012 as the bike-share program matured. That year-over-year growth is another reason time-series CV matters: random k-fold lets the model see 2012 growth during training on "2011 test" folds, which does not happen in deployment.

## Method primer: time-series cross-validation and why random folds leak

Cross-validation exists to estimate how a model will perform on unseen data. The default scikit-learn tool, `KFold(shuffle=True)`, partitions the dataset into k random subsets, trains on k-1 of them, validates on the held-out subset, and rotates through. That procedure is valid when observations are exchangeable — when any row is as likely to be the validation target as any other.

Hourly demand on a bike-share system is not exchangeable. A row from December 2012 carries information about the ridership level the program has reached by its second year, information that is causally downstream of every row in 2011. If a random fold places December 2012 in training and April 2011 in validation, the model is learning from the future to score on the past. Deploying that model means predicting hours that come after training, not a random mix. Bergmeir and Benítez (2012) frame the choice in those terms: random CV is valid only when the time dimension carries no information, and for any series with trend or seasonality it does.

`TimeSeriesSplit` enforces chronology. For fold k, training is everything up to timestamp T(k), validation is the chunk that comes next. Training grows each fold, the validation window slides forward. That matches deployment. The cost is higher variance on earlier folds where training is thin — an honest cost of evaluating honestly.

The leakage is visible even on a model that never fits parameters. The naive baseline here is `cnt.groupby(["weekday", "hour"]).mean()`, a lookup table. Under random k-fold every validation row sees a seasonal mean computed over the full 24 months, so residuals are small. Under time-series CV, earlier folds evaluate a lookup built on less data, so residuals grow. The baseline's CV score is worse under the honest split. That is the correct direction of movement; the random number was an artefact.

## The story: validation strategy

![Bar chart comparing random k-fold and time-series CV RMSLE for three models.](figures/cv-strategy-comparison.png)

Three models, two cross-validation strategies, six bars.

| Model | Random k-fold Log-RMSLE | Time-series 5-fold Log-RMSLE |
| --- | ---: | ---: |
| Naive seasonal mean (weekday × hour) | 0.644 | **0.509** |
| Ridge with cyclical features | 1.151 | 1.230 |
| LightGBM with seasonal features | **0.405** | 0.595 |

Under random k-fold, LightGBM wins by a wide margin. Under time-series CV, the naive seasonal baseline wins instead. LightGBM's time-series score is 46.6 percent worse than its k-fold score. The naive baseline goes the other way: its time-series score is 21 percent better than its k-fold score, because time-series CV happens to evaluate it mostly on the second year, where a seasonal mean built on year-one data is already stable.

Ridge sits at Log-RMSLE 1.151 under random k-fold and 1.230 under time-series. At Log-RMSLE 1.15 the typical multiplicative error factor is about 3.2 on back-transformed predictions — a 142-rentals hour comes out as roughly 45 or 450. Linear models with `sin(hour) + cos(hour)` cyclical features cannot bend hard enough to capture commuter-peak demand; a tree model pulls far more of that non-linearity for free.

## Why random k-fold leaks here

![Animation showing the five time-series folds being revealed, then contrasted with a random 5-fold strategy on the same timeline.](figures/timeseries-cv-animation.gif)

The animation builds the two strategies on the same 17,379-row timeline. Top panel: expanding-window TimeSeriesSplit, five folds whose validation windows slide forward. Bottom panel: random 5-fold, same data, same five validation sets, but now each fold's validation bars are scattered across every month. The fold-by-fold Log-RMSLE scoreboard updates as each fold arrives, so the difference between the strategies is visible as a number.

Random 5-fold ignores the time dimension. For a training fold containing 80 percent of the data drawn uniformly at random, the model sees observations from December 2012 while validating on April 2011 and vice versa. That is information leakage. The model is being told what the trend looks like in the future, which it uses to fit the validation fold better than it has any business doing. The net effect is that random k-fold overstates the generalisation quality of any time-structured model.

## What actually wins

Nothing, among the three models tested under honest validation. LightGBM has the best time-series Log-RMSLE of the trained models (0.595), but the naive baseline's 0.509 sits inside that — the base rate "what is the mean demand for this hour of this weekday" explains more variance than LightGBM's engineered features and weather recover on truly unseen future data.

The practical implication is that deploying LightGBM as a residual predictor on top of the naive baseline — model the difference between actual demand and the seasonal mean, then add the seasonal mean back at prediction time — usually works better than deploying LightGBM directly. That approach is not in this project, but it is the natural next step.

## What this isn't

Not a Kaggle-winning submission. The Kaggle Bike Sharing Demand leaderboard uses Log-RMSLE against a held-out test set from the same time window, so random sampling gives a reasonable approximation of what the public leaderboard measures. For that setting, the random score is the number that matters. This project is explicitly about the difference between that and real deployment.

Not a test of modern time-series models either. No Prophet, no LSTM, no Temporal Fusion Transformer. The point is not that neural networks cannot beat LightGBM here. The point is that any model bound for production needs time-respecting validation to know what score it will actually hit.

## References

Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*, 2(2-3), 113-127.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
