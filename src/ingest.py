# Fetch DE-LU hourly data from Energy Charts API
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from . import config

def _get(url, params):
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(60 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 3:
                raise e
            time.sleep(5)

def _month_ranges(start, end):
    for d in pd.date_range(start, end, freq="MS"):
        end_d = min(d + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        yield d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")

def _to_df(d, cols):
    idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True).tz_convert(config.LOCAL_TZ)
    return pd.DataFrame({"timestamp_utc": idx.tz_convert("UTC"), "timestamp_berlin": idx, **cols})

def run_ingestion():
    print("Fetching DA prices...")
    out = []
    for a, b in _month_ranges(config.START_DATE, config.END_DATE):
        d = _get(config.ENDPOINTS["day_ahead_price"]["url"], {"bzn": config.BIDDING_ZONE, "start": a, "end": b})
        if d.get("unix_seconds"):
            out.append(_to_df(d, {"price_eur_mwh": d["price"]}))
        time.sleep(2)
    prices = pd.concat(out, ignore_index=True).sort_values("timestamp_utc")
    print(f"  {len(prices)} price rows")

    print("Fetching wind/solar/load...")
    key_map = {config.WIND_ONSHORE_KEY: "wind_onshore_MW", config.WIND_OFFSHORE_KEY: "wind_offshore_MW",
               config.SOLAR_KEY: "solar_MW", config.LOAD_KEY: "load_MW"}
    out = []
    for a, b in _month_ranges(config.START_DATE, config.END_DATE):
        d = _get(config.ENDPOINTS["public_power"]["url"], {"country": config.COUNTRY, "start": a, "end": b})
        if d.get("unix_seconds"):
            idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True).tz_convert(config.LOCAL_TZ)
            df = pd.DataFrame({"timestamp_utc": idx.tz_convert("UTC"), "timestamp_berlin": idx})
            avail = {x["name"]: x["data"] for x in d.get("production_types", [])}
            for k, col in key_map.items():
                df[col] = avail.get(k, [np.nan] * len(idx))
            out.append(df)
        time.sleep(2)
    power = pd.concat(out, ignore_index=True).sort_values("timestamp_utc")
    print(f"  {len(power)} power rows")

    print("Fetching cross-border flows...")
    out = []
    for a, b in _month_ranges(config.START_DATE, config.END_DATE):
        d = _get(config.ENDPOINTS["cross_border_physical_flows"]["url"], {"country": config.COUNTRY, "start": a, "end": b})
        if d.get("unix_seconds"):
            idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True).tz_convert(config.LOCAL_TZ)
            net = np.zeros(len(d["unix_seconds"]))
            for c in d.get("countries", []):
                net += np.nan_to_num(np.array(c["data"]), nan=0)
            out.append(pd.DataFrame({"timestamp_utc": idx.tz_convert("UTC"), "timestamp_berlin": idx, "net_import_GW": net}))
        time.sleep(2)
    flows = pd.concat(out, ignore_index=True).sort_values("timestamp_utc")
    print(f"  {len(flows)} flow rows")

    p = prices.set_index("timestamp_utc")[["price_eur_mwh"]].resample("1h").mean()
    pw = power.set_index("timestamp_utc")[["wind_onshore_MW", "wind_offshore_MW", "solar_MW", "load_MW"]].resample("1h").mean()
    pf = flows.set_index("timestamp_utc")[["net_import_GW"]].resample("1h").mean()
    merged = p.join(pw, how="outer").join(pf, how="outer")
    merged["timestamp_berlin"] = merged.index.tz_convert(config.LOCAL_TZ)
    merged = merged[["timestamp_berlin", "price_eur_mwh", "wind_onshore_MW", "wind_offshore_MW", "solar_MW", "load_MW", "net_import_GW"]]
    merged.index.name = "timestamp_utc"
    print(f"  Merged {len(merged)} rows")
    return merged
