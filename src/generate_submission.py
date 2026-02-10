# submission.csv + 3 figures
import calendar
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from . import config

Path(config.FIGURES_DIR).mkdir(exist_ok=True)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#f8f9fa", "axes.grid": True, "grid.alpha": 0.3})

def main():
    print("Generating submission...")
    df = pd.read_csv(Path(config.DATA_DIR) / "forecast_predictions.csv")
    # Out-of-sample test window (defined in config): TEST_START to TEST_END, hourly UTC
    test_window = f"{config.TEST_START} to {config.TEST_END} (UTC, hourly)"
    out = df[["timestamp_utc"]].rename(columns={"timestamp_utc": "id"}).assign(y_pred=df["pred_lgbm"].round(2))
    with open("submission.csv", "w") as f:
        f.write(f"# Test window: {test_window}\n")
        out.to_csv(f, index=False)
    print(f"  submission.csv ({len(out)} rows)")
    print(f"  Test window: {test_window}")

    if (Path(config.REPORTS_DIR) / "qa_report.json").exists():
        r = json.loads((Path(config.REPORTS_DIR) / "qa_report.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Fig 1: Data quality – {config.BIDDING_ZONE}", fontweight="bold")
        months, counts = list(r["monthly"].keys()), list(r["monthly"].values())
        axes[0].bar(range(len(months)), counts, color="#2ecc71")
        axes[0].set_xticks(range(len(months))); axes[0].set_xticklabels([m[-5:] for m in months], rotation=45)
        axes[0].set_ylabel("Hours"); axes[0].set_title("Monthly coverage")
        axes[1].axis("off")
        po = r["outliers"].get("price_eur_mwh", {})
        tbl = [["Rows", str(r["actual_hours"])], ["Missing", str(r["missing_hours"])], ["Dups", str(r["duplicates"])], ["Coverage", f"{r['actual_hours']}/{r['expected_hours']}"], ["Outliers", f"{po.get('n_outliers', 0)} ({po.get('pct', 0)}%)"], ["Neg price", str(po.get("n_negative", 0))]]
        axes[1].table(cellText=tbl, colLabels=["Check", "Result"], loc="center")
        fig.savefig(Path(config.FIGURES_DIR) / "fig1_data_quality_summary.png", bbox_inches="tight"); plt.close()
        print("  fig1_data_quality_summary.png")

    df = pd.read_csv(Path(config.DATA_DIR) / "forecast_predictions.csv")
    df["timestamp_berlin"] = pd.to_datetime(df["timestamp_berlin"], utc=True).dt.tz_convert(config.LOCAL_TZ)
    df["_m"] = df["timestamp_berlin"].dt.month
    fc = json.loads((Path(config.REPORTS_DIR) / "forecast_results.json").read_text())
    test = fc["test_period"]
    months = sorted(df["_m"].unique())
    fig, axes = plt.subplots(len(months), 1, figsize=(12, 3.2 * len(months)))
    if len(months) == 1: axes = [axes]
    fig.suptitle(f"Fig 2: LightGBM vs actual ({test['start']}–{test['end']})", fontweight="bold")
    for ax, m in zip(axes, months):
        sub = df[df["_m"] == m]
        t = sub["timestamp_berlin"]
        ax.plot(t, sub["actual"], label="Actual", color="#2c3e50")
        ax.plot(t, sub["pred_lgbm"], label="LightGBM", color="#e74c3c")
        ax.set_ylabel("EUR/MWh"); ax.set_title(calendar.month_name[m]); ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d", tz=t.dt.tz))
    fig.savefig(Path(config.FIGURES_DIR) / "fig2_forecast_vs_actual.png", bbox_inches="tight"); plt.close()
    print("  fig2_forecast_vs_actual.png")

    m = fc["metrics"]
    names = list(m.keys())
    mae_vals = [m[n]["overall"]["MAE"] for n in names]
    rmse_vals = [m[n]["overall"]["RMSE"] for n in names]
    bias_vals = [m[n]["overall"]["bias"] for n in names]
    tail_mae_vals = [m[n].get("tail_MAE") or 0 for n in names]  # MAE on bottom 5% + top 5% price hours
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f"Fig 3: Model comparison ({test['start']}–{test['end']})", fontweight="bold")
    colors = ["#95a5a6", "#3498db", "#9b59b6", "#e74c3c"]
    axes[0].bar(names, mae_vals, color=colors); axes[0].set_title("MAE"); axes[0].set_ylabel("EUR/MWh")
    axes[1].bar(names, rmse_vals, color=colors); axes[1].set_title("RMSE"); axes[1].set_ylabel("EUR/MWh")
    axes[2].bar(names, bias_vals, color=["#e74c3c" if b > 0 else "#3498db" for b in bias_vals]); axes[2].set_title("Bias"); axes[2].set_ylabel("EUR/MWh"); axes[2].axhline(0, color="black", linestyle="--")
    axes[3].bar(names, tail_mae_vals, color=colors); axes[3].set_title("Tail MAE (5th–95th %ile)"); axes[3].set_ylabel("EUR/MWh")
    for ax in axes: ax.tick_params(axis="x", rotation=15)
    fig.savefig(Path(config.FIGURES_DIR) / "fig3_model_comparison.png", bbox_inches="tight"); plt.close()
    print("  fig3_model_comparison.png")
    print("Done.")

if __name__ == "__main__":
    main()
