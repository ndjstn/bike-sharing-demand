"""Bike-sharing demand with real time-series cross-validation.

UCI hourly bike-sharing dataset (17,379 observations spanning 2011-2012).
The story: random k-fold CV gives misleadingly optimistic scores. Proper
time-series CV lands where you'd actually end up if you deployed the model.
And a naive seasonal-mean baseline is surprisingly hard to beat.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _palette import BIKE_SHARING as P, apply_to_mpl  # noqa: E402

sns.set_style("whitegrid")
apply_to_mpl(P)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 11})


def _cmap_native():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("project", [P.bg, P.cover_subtitle, P.accent, P.header_bg])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--outputs", required=True)
    return ap.parse_args()


def rmsle(y, yhat):
    yhat = np.maximum(0, yhat)
    return float(np.sqrt(mean_squared_error(np.log1p(y), np.log1p(yhat))))


def main():
    args = parse_args()
    fig_dir, out_dir = Path(args.figures), Path(args.outputs)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    print(f"rows: {len(df):,}  range: {df['dteday'].min().date()} to {df['dteday'].max().date()}")

    # Hero-style figure: hour-weekday demand heatmap
    heat = df.groupby(["weekday", "hr"])["cnt"].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    sns.heatmap(heat, cmap=_cmap_native(), cbar_kws={"label": "Mean hourly rentals"},
                yticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], ax=ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Weekday")
    ax.set_title("Mean hourly rentals: weekday rush hours vs. weekend leisure peaks")
    fig.tight_layout()
    fig.savefig(fig_dir / "hour-weekday-heatmap.png")
    plt.close(fig)

    # Seasonal line: monthly cumulative demand
    monthly = df.groupby(["yr", "mnth"])["cnt"].sum().reset_index()
    monthly["label"] = np.where(monthly["yr"] == 0, "2011", "2012")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for yr, lab, color in [(0, "2011", P.muted), (1, "2012", P.accent)]:
        d = monthly.loc[monthly["yr"] == yr]
        ax.plot(d["mnth"], d["cnt"] / 1000, "o-", lw=2.5, ms=7, color=color, label=lab)
    ax.set_xlabel("Month")
    ax.set_ylabel("Rentals (thousands)")
    ax.set_title("Monthly rentals: strong seasonality and year-over-year growth")
    ax.set_xticks(range(1, 13))
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "monthly-seasonality.png")
    plt.close(fig)

    # Temperature effect
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = pd.cut(df["temp"], bins=10)
    temp_g = df.groupby(bins, observed=True)["cnt"].mean()
    ax.bar(range(len(temp_g)), temp_g.values, color=P.accent)
    ax.set_xticks(range(len(temp_g)))
    ax.set_xticklabels([f"{c.mid:.2f}" for c in temp_g.index], rotation=45)
    ax.set_xlabel("Normalised temperature (midpoint of bin)")
    ax.set_ylabel("Mean hourly rentals")
    ax.set_title("Demand rises with temperature and falls off only above 0.85 norm.")
    fig.tight_layout()
    fig.savefig(fig_dir / "temperature-effect.png")
    plt.close(fig)

    # Feature engineering
    df["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    df["day_of_year"] = df["dteday"].dt.dayofyear
    df["days_since_start"] = (df["dteday"] - df["dteday"].min()).dt.days

    features = [
        "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
        "temp", "atemp", "hum", "windspeed",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "days_since_start",
    ]
    X = df[features].astype(float)
    y = df["cnt"].values

    # Baseline: naive seasonal mean by (weekday, hour)
    season_mean = df.groupby(["weekday", "hr"])["cnt"].mean()

    def naive_predict(idx_rows):
        keys = list(zip(df.loc[idx_rows, "weekday"], df.loc[idx_rows, "hr"]))
        return season_mean.reindex(keys).values

    # Two CV strategies: random KFold and TimeSeriesSplit
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tss = TimeSeriesSplit(n_splits=5)

    def cv_score(model_factory, folds, x_data, y_data):
        scores = []
        for tr, te in folds.split(x_data):
            m = model_factory()
            m.fit(x_data.iloc[tr], y_data[tr])
            yhat = m.predict(x_data.iloc[te])
            scores.append(rmsle(y_data[te], yhat))
        return np.array(scores)

    def naive_score(folds, x_data, y_data):
        scores = []
        for _, te in folds.split(x_data):
            yhat = naive_predict(te)
            yhat = np.where(np.isnan(yhat), y_data.mean(), yhat)
            scores.append(rmsle(y_data[te], yhat))
        return np.array(scores)

    def lgbm(): return LGBMRegressor(n_estimators=600, num_leaves=31, learning_rate=0.05, verbose=-1, random_state=42)
    def ridge(): return Ridge(alpha=1.0, random_state=42)

    scores = {
        ("naive_seasonal", "random_kfold"): naive_score(kf, X, y),
        ("naive_seasonal", "time_series"):  naive_score(tss, X, y),
        ("ridge",          "random_kfold"): cv_score(ridge, kf, X, y),
        ("ridge",          "time_series"):  cv_score(ridge, tss, X, y),
        ("lightgbm",       "random_kfold"): cv_score(lgbm, kf, X, y),
        ("lightgbm",       "time_series"):  cv_score(lgbm, tss, X, y),
    }

    summary_rows = []
    for (model, strategy), vals in scores.items():
        summary_rows.append({"model": model, "cv_strategy": strategy,
                              "mean_rmsle": round(float(vals.mean()), 4),
                              "std_rmsle": round(float(vals.std()), 4)})
    comparison = pd.DataFrame(summary_rows)
    print(comparison.to_string(index=False))
    comparison.to_csv(out_dir / "cv_comparison.csv", index=False)

    # The central figure: CV-strategy-comparison bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.35
    models = ["naive_seasonal", "ridge", "lightgbm"]
    kf_means = [scores[(m, "random_kfold")].mean() for m in models]
    tss_means = [scores[(m, "time_series")].mean() for m in models]
    kf_std = [scores[(m, "random_kfold")].std() for m in models]
    tss_std = [scores[(m, "time_series")].std() for m in models]
    idx = np.arange(len(models))
    ax.bar(idx - w/2, kf_means, w, yerr=kf_std, color=P.muted, label="Random 5-fold", capsize=4, edgecolor=P.bg, linewidth=1)
    ax.bar(idx + w/2, tss_means, w, yerr=tss_std, color=P.accent, label="Time-series 5-fold", capsize=4, edgecolor=P.bg, linewidth=1)
    for i, (km, tm) in enumerate(zip(kf_means, tss_means)):
        ax.text(i - w/2, km, f"{km:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, tm, f"{tm:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(idx); ax.set_xticklabels(["Naive seasonal", "Ridge", "LightGBM"])
    ax.set_ylabel("RMSLE (lower is better)")
    ax.set_title("Validation strategy matters more than model choice here")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "cv-strategy-comparison.png")
    plt.close(fig)

    # Animation: walk through the time-series folds visually
    n_splits = 5
    tss_anim = TimeSeriesSplit(n_splits=n_splits)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, len(X)); ax.set_ylim(-0.5, n_splits - 0.5)
    ax.set_xlabel("Observation index (ordered by time)")
    ax.set_ylabel("Fold")
    ax.set_title("Time-series 5-fold: train grows, validation slides forward")
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"fold {i+1}" for i in range(n_splits)])

    fold_patches = []
    for fold_idx, (tr, te) in enumerate(tss_anim.split(X)):
        train_rect = plt.Rectangle((tr[0], fold_idx - 0.35), tr[-1] - tr[0], 0.7, color=P.muted, alpha=0)
        val_rect = plt.Rectangle((te[0], fold_idx - 0.35), te[-1] - te[0], 0.7, color=P.accent, alpha=0)
        ax.add_patch(train_rect); ax.add_patch(val_rect)
        fold_patches.append((train_rect, val_rect))

    ax.legend(handles=[plt.Rectangle((0,0),1,1,color=P.muted,alpha=0.85, label="Train"),
                        plt.Rectangle((0,0),1,1,color=P.accent,alpha=0.85, label="Validate")],
              loc="upper right")

    def animate(i):
        # Reveal folds one at a time
        for j, (tr, va) in enumerate(fold_patches):
            alpha_tr = 0.85 if j <= i else 0.0
            alpha_va = 0.9 if j <= i else 0.0
            tr.set_alpha(alpha_tr); va.set_alpha(alpha_va)
        return [p for pair in fold_patches for p in pair]

    anim = animation.FuncAnimation(fig, animate, frames=n_splits, interval=900, blit=False)
    anim.save(str(fig_dir / "timeseries-cv-animation.gif"), writer="pillow", fps=1)
    plt.close(fig)

    summary = {
        "rows": int(len(df)),
        "date_range": [str(df["dteday"].min().date()), str(df["dteday"].max().date())],
        "cv_comparison": summary_rows,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# Bike-sharing demand summary", ""]
    md.append(f"Rows: {len(df):,}. Date range: {df['dteday'].min().date()} to {df['dteday'].max().date()}.")
    md.append("")
    md.append("## CV strategy comparison (5 folds, RMSLE)")
    md.append("")
    md.append("| Model | Random k-fold | Time-series CV |")
    md.append("|---|---:|---:|")
    for m in models:
        kf_v = scores[(m, "random_kfold")].mean()
        tss_v = scores[(m, "time_series")].mean()
        md.append(f"| {m} | {kf_v:.4f} | {tss_v:.4f} |")
    (out_dir / "analysis_summary.md").write_text("\n".join(md))
    print("Done")


if __name__ == "__main__":
    main()
