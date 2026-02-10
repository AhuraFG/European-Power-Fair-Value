# QA narrative (from computed metrics only) + drivers commentary. Ollama local.
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from langchain_openai import ChatOpenAI
from . import config

def get_llm():
    return ChatOpenAI(base_url=config.LLM_BASE_URL, api_key="ollama", model=config.LLM_MODEL,
                      temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS)

def compute_dataset_stats(df):
    n = len(df)
    out = {"n_rows": n, "numeric": {}}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        out["numeric"][col] = {
            "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
            "p1": round(float(s.quantile(0.01)), 4), "p99": round(float(s.quantile(0.99)), 4),
            "pct_negative": round(100 * (s < 0).sum() / n, 2), "pct_zero": round(100 * (s == 0).sum() / n, 2),
        }
    return out

def load_baseline_summary():
    p = Path(config.REPORTS_DIR) / "qa_report.json"
    return json.loads(p.read_text()) if p.exists() else None

def compute_narrative_metrics(df, baseline):
    n = len(df)
    out = {"n_rows": n}
    if baseline:
        out["expected_hours"] = baseline.get("expected_hours")
        out["actual_hours"] = baseline.get("actual_hours")
        out["missing_hours"] = baseline.get("missing_hours", 0)
        out["duplicates"] = baseline.get("duplicates", 0)
        out["all_timestamps_present"] = out["missing_hours"] == 0 and out["duplicates"] == 0
        dst = baseline.get("dst", [])
        out["dst_ok"] = all(d.get("ok", False) for d in dst) if dst else True
        out["dst_dates"] = [d.get("date") for d in dst]
        po = (baseline.get("outliers") or {}).get("price_eur_mwh", {})
        out["price_n_negative"] = po.get("n_negative", 0)
        out["price_pct_negative"] = round(100 * out["price_n_negative"] / n, 1) if n else 0
        out["price_outlier_pct"] = po.get("pct")
    else:
        out["all_timestamps_present"] = n > 0
        out["dst_ok"] = None
        if "price_eur_mwh" in df.columns:
            neg = (df["price_eur_mwh"] < 0).sum()
            out["price_n_negative"] = int(neg)
            out["price_pct_negative"] = round(100 * neg / n, 1) if n else 0
    if "net_import_GW" in df.columns:
        s = df["net_import_GW"]
        run_len = s.groupby((s != s.shift()).cumsum()).size()
        out["net_import_flatline_max_hours"] = int(run_len.max()) if len(run_len) else 0
    else:
        out["net_import_flatline_max_hours"] = 0
    # Volatility: last 7 days vs prior 7 days. Slice by position first so date ranges match the std windows (no dropna before slice).
    if "price_eur_mwh" in df.columns and n >= 24 * 14:
        tz_berlin = df.index.tz_convert(config.LOCAL_TZ)
        last_7d = df["price_eur_mwh"].iloc[-24 * 7 :]
        prior_7d = df["price_eur_mwh"].iloc[-24 * 14 : -24 * 7]
        out["volatility_window_current"] = f"{tz_berlin[-24*7].date()} to {tz_berlin[-1].date()}"
        out["volatility_window_prior"] = f"{tz_berlin[-24*14].date()} to {tz_berlin[-24*7 - 1].date()}"
        std_this = float(last_7d.std())
        std_prev = float(prior_7d.std())
        out["volatility_std_current"] = round(std_this, 2) if pd.notna(std_this) else None
        out["volatility_std_prior"] = round(std_prev, 2) if pd.notna(std_prev) else None
        if std_prev and std_prev > 0 and pd.notna(std_this):
            out["volatility_pct_change"] = round(100 * (std_this - std_prev) / std_prev, 1)
        else:
            out["volatility_pct_change"] = None
    else:
        out["volatility_window_current"] = None
        out["volatility_window_prior"] = None
        out["volatility_std_current"] = None
        out["volatility_std_prior"] = None
        out["volatility_pct_change"] = None
    return out

def run_llm_qa_and_save_cleaned():
    path = Path(config.DATA_DIR) / config.BASELINE_PARQUET
    if not path.exists():
        print("  LLM QA skipped: no baseline (run Task 1 first)")
        return
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    baseline = load_baseline_summary()
    stats = compute_dataset_stats(df)
    metrics = compute_narrative_metrics(df, baseline)
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    df.to_parquet(Path(config.DATA_DIR) / "de_hourly_dataset.parquet")
    print(f"  Dataset (baseline only) → data/de_hourly_dataset.parquet ({len(df)} rows)")
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    with open(Path(config.REPORTS_DIR) / "qa_llm_report.json", "w") as f:
        json.dump({"narrative_metrics": metrics, "dataset_stats": stats}, f, indent=2, default=str)
    # QA narrative: single schema = date-range only (volatility_window_* / volatility_std_* / volatility_pct_change)
    prompt = f"""Using ONLY these numbers, write a short QA report (few bullet lines). Do not invent figures.
For volatility, use the exact date ranges given: e.g. "In [volatility_window_current], price volatility (std) was [volatility_std_current] vs [volatility_std_prior] in [volatility_window_prior] ([volatility_pct_change]%)."
Metrics: {json.dumps(metrics, indent=2)}
Baseline: {json.dumps(baseline or {}, indent=2)[:1200]}
Output only the report lines."""
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    out_path = Path(config.REPORTS_DIR) / "qa_narrative.txt"
    log = {"model": config.LLM_MODEL, "ts_utc": datetime.now(timezone.utc).isoformat(), "prompt": prompt[:3000], "output": None, "error": None, "latency_s": None}
    t0 = time.perf_counter()
    try:
        out = get_llm().invoke(prompt).content
        log["latency_s"] = round(time.perf_counter() - t0, 2)
        log["output"] = out
        out_path.write_text(out, encoding="utf-8")
        print(f"  QA narrative → reports/qa_narrative.txt")
    except Exception as e:
        log["error"] = str(e)
        log["latency_s"] = round(time.perf_counter() - t0, 2)
        lines = []
        if metrics.get("all_timestamps_present"):
            lines.append("All timestamps present.")
        if metrics.get("dst_ok") is True:
            lines.append("DST ok.")
        if metrics.get("price_pct_negative") is not None:
            lines.append(f"Price has {metrics['price_pct_negative']}% negative hours.")
        if metrics.get("net_import_flatline_max_hours", 0) > 1:
            lines.append(f"Net imports flatlined for {metrics['net_import_flatline_max_hours']} hours.")
        if metrics.get("volatility_pct_change") is not None and metrics.get("volatility_window_current"):
            w_cur, w_prior = metrics["volatility_window_current"], metrics["volatility_window_prior"]
            s_cur, s_prior = metrics.get("volatility_std_current"), metrics.get("volatility_std_prior")
            pct = metrics["volatility_pct_change"]
            lines.append(f"In {w_cur}, price volatility (std) was {s_cur} vs {s_prior} in {w_prior} ({pct}%).")
        fallback = "\n".join(lines) if lines else f"QA (no API). Rows: {metrics.get('n_rows')}."
        log["output"] = fallback
        out_path.write_text(fallback, encoding="utf-8")
        print(f"  QA narrative (fallback) → reports/qa_narrative.txt")
    with open(Path(config.REPORTS_DIR) / "qa_narrative_log.json", "w") as f:
        json.dump(log, f, indent=2)

def run_drivers_commentary():
    r = Path(config.REPORTS_DIR)
    fc_path, curve_path = r / "forecast_results.json", r / "curve_trading_sheet.json"
    if not fc_path.exists():
        Path(config.REPORTS_DIR).mkdir(exist_ok=True)
        (Path(config.REPORTS_DIR) / "ai_commentary.txt").write_text("Run forecast and curve first.")
        return
    fc = json.loads(fc_path.read_text())
    curve = json.loads(curve_path.read_text()) if curve_path.exists() else {}
    models = fc.get("metrics", {})
    perf = curve.get("signal_performance", {})
    recent = curve.get("signals", [])[-5:]
    prompt = f"""Short daily note (2-3 paras). Use ONLY these numbers; cite source in parentheses.
Forecast: {json.dumps(models, indent=2)}
Signals: {json.dumps(perf, indent=2)}
Recent: {json.dumps(recent, indent=2)}
End with: Sources: reports/forecast_results.json, reports/curve_trading_sheet.json"""
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    log = {"model": config.LLM_MODEL, "ts_utc": datetime.now(timezone.utc).isoformat(), "prompt": prompt[:3000], "output": None, "error": None, "latency_s": None}
    t0 = time.perf_counter()
    try:
        out = get_llm().invoke(prompt).content
        log["latency_s"] = round(time.perf_counter() - t0, 2)
        log["output"] = out
    except Exception as e:
        log["error"] = str(e)
        log["latency_s"] = round(time.perf_counter() - t0, 2)
        best = min(models, key=lambda n: models[n]["overall"]["MAE"])
        out = f"No API. Best: {best} MAE {models[best]['overall']['MAE']}. Hit rate {perf.get('hit_rate', 0)}%."
        log["output"] = out
    (Path(config.REPORTS_DIR) / "ai_commentary.txt").write_text(out)
    with open(r / "ai_commentary_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(out[:280] + "..." if len(out) > 280 else out)

def main():
    print("Task 4: AI commentary")
    run_drivers_commentary()
    print("Done.")

if __name__ == "__main__":
    main()
