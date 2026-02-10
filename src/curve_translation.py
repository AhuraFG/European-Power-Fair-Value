# Forecast → delivery means, uncertainty bands, signals
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import config

def main():
    print("Task 3: Curve translation")
    df = pd.read_csv(Path(config.DATA_DIR) / "forecast_predictions.csv")
    df["timestamp_berlin"] = pd.to_datetime(df["timestamp_berlin"], utc=True).dt.tz_convert(config.LOCAL_TZ)
    df["date"] = df["timestamp_berlin"].dt.date
    df["month"] = df["timestamp_berlin"].dt.month

    daily = {str(d): {"base_fc": round(g["pred_lgbm"].mean(), 2), "base_act": round(g["actual"].mean(), 2)} for d, g in df.groupby("date")}
    monthly = {str(m): {"base_fc": round(g["pred_lgbm"].mean(), 2), "base_act": round(g["actual"].mean(), 2)} for m, g in df.groupby("month")}

    df["err"] = df["actual"] - df["pred_lgbm"]
    bands = {}
    for d in sorted(df["date"].unique()):
        past = df[pd.to_datetime(df["date"]) < pd.Timestamp(d)].tail(30 * 24)
        if len(past) >= 48:
            fc = df[df["date"] == d]["pred_lgbm"].mean()
            e = past["err"]
            bands[str(d)] = {"P10": round(fc + e.quantile(0.1), 2), "P90": round(fc + e.quantile(0.9), 2)}

    dates = sorted(df["date"].unique())
    sigs = []
    for i in range(1, len(dates)):
        today = df[df["date"] == dates[i]]
        yest = df[df["date"] == dates[i-1]]
        if len(today) < 20 or len(yest) < 20:
            continue
        anchor = yest["actual"].mean()
        fc, act = today["pred_lgbm"].mean(), today["actual"].mean()
        z = (fc - anchor) / max(yest["actual"].std(), 1)
        agree = np.mean([1 if today[c].mean() > anchor else -1 for c in ["pred_naive", "pred_ridge", "pred_rf", "pred_lgbm"]])
        score = 0.4 * z + 0.35 * agree + 0.25
        sig = "BUY" if score > 0.5 else "SELL" if score < -0.5 else "NEUTRAL"
        correct = (sig == "BUY" and act > anchor) or (sig == "SELL" and act < anchor) if sig != "NEUTRAL" else None
        pnl = (act - anchor) if sig == "BUY" else (anchor - act) if sig == "SELL" else None
        sigs.append({"date": str(dates[i]), "signal": sig, "forecast": round(fc, 2), "anchor": round(anchor, 2), "actual": round(act, 2), "correct": correct, "pnl": round(pnl, 2) if pnl is not None else None})

    directed = [s for s in sigs if s["signal"] != "NEUTRAL"]
    correct = sum(1 for s in directed if s["correct"])
    hit = 100 * correct / len(directed) if directed else 0
    print(f"  {len(directed)} directed signals, {correct} correct ({hit:.1f}% hit)")
    # Forward expected averages (from forecast_results: next day/week/month from forecast distribution)
    forward = {}
    fc_path = Path(config.REPORTS_DIR) / "forecast_results.json"
    if fc_path.exists():
        fc = json.loads(fc_path.read_text())
        forward = fc.get("forward_expected_averages") or {}
    report = {
        "delivery_periods": {"daily": daily, "monthly": monthly},
        "uncertainty_bands": bands,
        "signals": sigs,
        "signal_performance": {"directed": len(directed), "correct": correct, "hit_rate": round(hit, 1)},
        "forward_expected_averages": forward,
    }
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    with open(Path(config.REPORTS_DIR) / "curve_trading_sheet.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    lines = [f"Curve – {config.BIDDING_ZONE}", f"Backtest: {hit:.1f}% hit on {len(directed)} directed."]
    if forward.get("next_day_expected_avg") is not None:
        lines.append(f"Next day ({forward.get('next_day_date')}) expected avg: {forward['next_day_expected_avg']} EUR/MWh")
    if forward.get("next_week_expected_avg") is not None:
        lines.append(f"Next week ({forward.get('next_week_dates')}) expected avg: {forward['next_week_expected_avg']} EUR/MWh")
    if forward.get("next_month_expected_avg") is not None:
        lines.append(f"Next month ({forward.get('next_month_label')}) expected avg: {forward['next_month_expected_avg']} EUR/MWh")
    with open(Path(config.REPORTS_DIR) / "curve_trading_sheet.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Done.")

if __name__ == "__main__":
    main()
