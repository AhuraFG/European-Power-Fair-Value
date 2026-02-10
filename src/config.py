# DE-LU pipeline. API: https://api.energy-charts.info
import os

API_BASE = "https://api.energy-charts.info"
COUNTRY, BIDDING_ZONE = "de", "DE-LU"
START_DATE, END_DATE = "2025-01-01", "2025-12-31"
LOCAL_TZ = "Europe/Berlin"
WIND_ONSHORE_KEY, WIND_OFFSHORE_KEY = "Wind onshore", "Wind offshore"
SOLAR_KEY, LOAD_KEY = "Solar", "Load"

DATA_DIR, REPORTS_DIR, FIGURES_DIR = "data", "reports", "figures"
BASELINE_PARQUET = "de_hourly_baseline.parquet"

TEST_START, TEST_END = "2025-10-01", "2025-12-31"
PEAK_START, PEAK_END = 8, 20

ENDPOINTS = {
    "day_ahead_price": {"url": f"{API_BASE}/price"},
    "public_power": {"url": f"{API_BASE}/public_power"},
    "cross_border_physical_flows": {"url": f"{API_BASE}/cbpf"},
}

# Ollama (run: ollama run qwen2.5:7b)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "3000"))
