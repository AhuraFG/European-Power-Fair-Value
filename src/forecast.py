# Walk-forward forecast: Naive, Ridge, RF, LightGBM
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import config
from . import features as feat
from .models import SeasonalNaive, RidgeModel, RFModel, LGBMModel

def load_data():
    # Prefer dataset (written by run_llm_qa_and_save_cleaned); fallback to baseline (written by Task 1) so Task 2 works after Task 1 only
    data_dir = Path(config.DATA_DIR)
    path = data_dir / "de_hourly_dataset.parquet"
    if not path.exists():
        path = data_dir / config.BASELINE_PARQUET
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return feat.build(df)

def walk_forward(df):
    start = pd.Timestamp(config.TEST_START, tz=config.LOCAL_TZ)
    end = pd.Timestamp(config.TEST_END, tz=config.LOCAL_TZ) + pd.Timedelta(days=1)
    mask = (df["timestamp_berlin"] >= start) & (df["timestamp_berlin"] < end)
    days = sorted(df.loc[mask, "timestamp_berlin"].dt.normalize().unique())
    models = [SeasonalNaive(), RidgeModel(), RFModel(), LGBMModel()]
    cols = ["pred_naive", "pred_ridge", "pred_rf", "pred_lgbm"]
    rows = []
    last = -7
    dropped_test_hours = 0
    for i, day in enumerate(days):
        test_raw = df[df["timestamp_berlin"].dt.normalize() == day]
        test = test_raw.dropna(subset=feat.COLS_FULL + [feat.TARGET])
        dropped_test_hours += len(test_raw) - len(test)
        if len(test) == 0:
            continue
        train = df[df["timestamp_berlin"] < day].dropna(subset=feat.COLS_FULL + [feat.TARGET])
        if len(train) < 720:
            continue
        if i - last >= 7:
            for m in models:
                m.fit(train, train[feat.TARGET])
            last = i
        preds = {c: models[j].predict(test) for j, c in enumerate(cols)}
        for k, (idx, r) in enumerate(test.iterrows()):
            rows.append({"timestamp_utc": idx, "timestamp_berlin": r["timestamp_berlin"], "hour_berlin": r["timestamp_berlin"].hour,
                         "actual": r[feat.TARGET], **{c: preds[c][k] for c in cols}})
    if dropped_test_hours:
        print(f"  Dropped {dropped_test_hours} test hours (NaN in features or target)")
    return pd.DataFrame(rows)

def forward_expected_averages(df, res):
    """Derive next-day, next-week, next-month expected average from forecast (and one-day-ahead forward)."""
    out = {"next_day_expected_avg": None, "next_day_date": None,
           "next_week_expected_avg": None, "next_week_dates": None,
           "next_month_expected_avg": None, "next_month_label": None}
    # Next week / next month from test-window forecast distribution (last 7 days, last calendar month)
    if res is not None and not res.empty and "pred_lgbm" in res.columns:
        res = res.copy()
        res["date"] = pd.to_datetime(res["timestamp_berlin"], utc=True).dt.tz_convert(config.LOCAL_TZ).dt.date
        dates = sorted(res["date"].unique())
        if len(dates) >= 7:
            last_7_dates = dates[-7:]
            last_7 = res[res["date"].isin(last_7_dates)]
            out["next_week_expected_avg"] = round(float(last_7["pred_lgbm"].mean()), 2)
            out["next_week_dates"] = f"{min(last_7_dates)} to {max(last_7_dates)}"
        last_month = res["date"].max()
        if pd.notna(last_month):
            month_df = res[res["date"].apply(lambda d: d.month == last_month.month and d.year == last_month.year)]
            if len(month_df) > 0:
                out["next_month_expected_avg"] = round(float(month_df["pred_lgbm"].mean()), 2)
                out["next_month_label"] = last_month.strftime("%Y-%m")
    # One-day-ahead forward: train on full data, predict next calendar day
    if df is not None and len(df) >= 720 and feat.TARGET in df.columns:
        train = df.dropna(subset=feat.COLS_FULL + [feat.TARGET])
        if len(train) < 720:
            return out
        last_utc = df.index.max()
        next_day_start = last_utc + pd.Timedelta(hours=1)
        new_idx = pd.date_range(next_day_start, periods=24, freq="h", tz="UTC")
        new_df = pd.DataFrame(index=new_idx)
        new_df["timestamp_berlin"] = new_df.index.tz_convert(config.LOCAL_TZ)
        for c in df.columns:
            if c not in new_df.columns:
                new_df[c] = np.nan
        extended = pd.concat([df, new_df]).sort_index()
        extended["timestamp_berlin"] = pd.to_datetime(extended["timestamp_berlin"], utc=True).dt.tz_convert(config.LOCAL_TZ)
        extended = feat.build(extended)
        next_24 = extended.iloc[-24:]
        if next_24[feat.COLS_FULL].notna().all().all():
            models = [SeasonalNaive(), RidgeModel(), RFModel(), LGBMModel()]
            for m in models:
                m.fit(train, train[feat.TARGET])
            preds = models[3].predict(next_24)  # LightGBM
            out["next_day_expected_avg"] = round(float(np.mean(preds)), 2)
            out["next_day_date"] = next_24["timestamp_berlin"].iloc[0].strftime("%Y-%m-%d")
    return out


def metrics(res):
    y = res["actual"].values
    # Tail = bottom 5% + top 5% of price hours (spikes are the hardest regime)
    q5, q95 = np.nanpercentile(y, 5), np.nanpercentile(y, 95)
    tail = (y <= q5) | (y >= q95)
    out = {}
    for col, name in [("pred_naive", "Seasonal Naive"), ("pred_ridge", "Ridge Regression"), ("pred_rf", "Random Forest"), ("pred_lgbm", "LightGBM")]:
        p = res[col].values
        ok = ~(np.isnan(y) | np.isnan(p))
        mae = lambda m: round(float(np.mean(np.abs(y[ok&m] - p[ok&m]))), 2) if (ok&m).sum() else None
        rmse = lambda m: round(float(np.sqrt(np.mean((y[ok&m]-p[ok&m])**2))), 2) if (ok&m).sum() else None
        bias = lambda m: round(float(np.mean(p[ok&m] - y[ok&m])), 2) if (ok&m).sum() else None
        out[name] = {"overall": {"MAE": mae(np.ones(len(y), dtype=bool)), "RMSE": rmse(np.ones(len(y), dtype=bool)), "bias": bias(np.ones(len(y), dtype=bool))}, "tail_MAE": mae(tail)}
    return out

def main():
    print("Task 2: Forecasting")
    df = load_data()
    res = walk_forward(df)
    if res.empty or "actual" not in res.columns:
        m = {n: {"overall": {"MAE": None, "RMSE": None, "bias": None}, "tail_MAE": None} for n in ["Seasonal Naive", "Ridge Regression", "Random Forest", "LightGBM"]}
        print(f"  {len(res)} predictions (no test data)")
    else:
        m = metrics(res)
        print(f"  {len(res)} predictions")
    for name in m:
        o = m[name]["overall"]
        print(f"  {name}: MAE {o['MAE']} RMSE {o['RMSE']} bias {o['bias']}")
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    out = res.copy() if not res.empty else pd.DataFrame(columns=["timestamp_utc", "timestamp_berlin", "hour_berlin", "actual", "pred_naive", "pred_ridge", "pred_rf", "pred_lgbm"])
    if not out.empty:
        out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        out["timestamp_berlin"] = out["timestamp_berlin"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.to_csv(Path(config.DATA_DIR) / "forecast_predictions.csv", index=False)
    forward = forward_expected_averages(df, res if not res.empty else None)
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    with open(Path(config.REPORTS_DIR) / "forecast_results.json", "w") as f:
        json.dump({
            "test_period": {"start": config.TEST_START, "end": config.TEST_END},
            "metrics": m,
            "forward_expected_averages": forward,
        }, f, indent=2, default=str)
    if forward.get("next_day_expected_avg") is not None:
        print(f"  Next day ({forward.get('next_day_date')}) expected avg: {forward['next_day_expected_avg']} EUR/MWh")
    if forward.get("next_week_expected_avg") is not None:
        print(f"  Next week ({forward.get('next_week_dates')}) expected avg: {forward['next_week_expected_avg']} EUR/MWh")
    if forward.get("next_month_expected_avg") is not None:
        print(f"  Next month ({forward.get('next_month_label')}) expected avg: {forward['next_month_expected_avg']} EUR/MWh")
    print("Done.")

if __name__ == "__main__":
    main()
