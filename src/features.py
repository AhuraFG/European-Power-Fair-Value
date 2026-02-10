# Features for DA price forecast. All inputs lagged >= 24h (known at auction, no same-day look-ahead).
import numpy as np
import pandas as pd

def build(df):
    df = df.copy()
    t = df["timestamp_berlin"]
    df["hour"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    p = df["price_eur_mwh"]
    df["price_lag_24h"] = p.shift(24)
    df["price_lag_168h"] = p.shift(168)
    df["price_rmean_7d"] = p.shift(24).rolling(168, min_periods=48).mean()
    df["price_rstd_7d"] = p.shift(24).rolling(168, min_periods=48).std()
    # Fundamentals lagged 24h (same hour D-1) so prediction for D uses only D-1 and earlier.
    df["wind_onshore_MW_lag24"] = df["wind_onshore_MW"].shift(24)
    df["wind_offshore_MW_lag24"] = df["wind_offshore_MW"].shift(24)
    df["solar_MW_lag24"] = df["solar_MW"].shift(24)
    df["load_MW_lag24"] = df["load_MW"].shift(24)
    df["net_import_GW_lag24"] = df["net_import_GW"].shift(24)
    df["wind_total_MW"] = df["wind_onshore_MW_lag24"] + df["wind_offshore_MW_lag24"]
    df["ren_penetration"] = ((df["wind_total_MW"] + df["solar_MW_lag24"]) / df["load_MW_lag24"]).clip(0, 3)
    return df

# Calendar + fundamentals (all lagged 24h for realism).
COLS_LINEAR = ["hour", "day_of_week", "month", "is_weekend",
               "wind_onshore_MW_lag24", "wind_offshore_MW_lag24", "solar_MW_lag24", "load_MW_lag24", "net_import_GW_lag24",
               "wind_total_MW", "ren_penetration"]
# Ridge and LightGBM: add price lags and 7d rolling (level anchor + shape).
COLS_FULL = COLS_LINEAR + ["price_lag_24h", "price_lag_168h", "price_rmean_7d", "price_rstd_7d"]
TARGET = "price_eur_mwh"
