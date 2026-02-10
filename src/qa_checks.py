# Baseline QA: missingness, duplicates, coverage, outliers, DST
import json
from pathlib import Path
import pandas as pd
import pytz
from . import config

def run_qa(df):
    n = len(df)
    miss = {c: {"missing": int(df[c].isna().sum()), "pct": round(100*df[c].isna().sum()/n, 2)} for c in df.columns}
    dups = int(df.index.duplicated(keep=False).sum())
    idx = df.index.sort_values()
    exp = pd.date_range(idx.min(), idx.max(), freq="1h", tz="UTC")
    missing_hrs = len(exp) - len(idx)
    monthly = df.groupby(df.index.to_period("M")).size().to_dict()

    outliers = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lo, hi = q1 - 3*(q3-q1), q3 + 3*(q3-q1)
        outliers[col] = {"n_outliers": int(((s<lo)|(s>hi)).sum()), "pct": round(100*((s<lo)|(s>hi)).sum()/len(s), 2)}
        if col == "price_eur_mwh":
            outliers[col]["n_negative"] = int((s < 0).sum())

    tz = pytz.timezone(config.LOCAL_TZ)
    berlin = df.index.tz_convert(config.LOCAL_TZ)
    dst = []
    for y in set(berlin.year):
        for tt in tz._utc_transition_times:
            if tt.year != y:
                continue
            d = pd.Timestamp(tt, tz="UTC").tz_convert(config.LOCAL_TZ).normalize()
            h = int((berlin.normalize() == d).sum())
            dst.append({"date": str(d.date()), "hours": h, "ok": h == (23 if tt.month <= 6 else 25)})

    report = {"rows": n, "missingness": miss, "duplicates": dups, "expected_hours": len(exp), "actual_hours": len(idx),
              "missing_hours": missing_hrs, "monthly": {str(k): int(v) for k, v in monthly.items()}, "outliers": outliers, "dst": dst}
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    with open(Path(config.REPORTS_DIR) / "qa_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  QA report → {config.REPORTS_DIR}/qa_report.json")
    print(f"  Rows {report['actual_hours']}/{report['expected_hours']}, missing {report['missing_hours']}, dups {dups}")
    for chk in dst:
        print(f"  DST {chk['date']}: {chk['hours']}h {'ok' if chk['ok'] else 'fail'}")
    return report
