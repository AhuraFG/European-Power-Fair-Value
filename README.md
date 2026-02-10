# DE-LU Power Market Pipeline

Student project: data ingestion, forecasting, curve translation and AI commentary for the German day-ahead power market.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

On macOS you may need: `brew install libomp` (for LightGBM).

## Run

From the **repo root**. The code lives in the `src` package and uses relative imports, so run it as a module (do not run `python src/forecast.py` or scripts directly):

```bash
# Run everything
python run_all.py

# Or step by step (python -m src.<module> from repo root)
python -m src.main              # 1. Ingest data + QA
python -m src.forecast         # 2. Forecast (needs Task 1; loads baseline or dataset)
python -m src.curve_translation # 3. Tradable view + signals
python -m src.ai_commentary    # 4. AI commentary (Ollama local only)
python -m src.generate_submission  # submission.csv + 3 figures
```

## Data & endpoints

- **Source**: [Energy Charts API](https://api.energy-charts.info/) (public).
- **Endpoints**: `/price?bzn=DE-LU` (DA prices), `/public_power?country=de` (wind, solar, load), `/cbpf?country=de` (cross-border flows).
- **Market**: DE-LU, hourly, 2025. Timezone: UTC → Europe/Berlin (DST handled).

## Ollama (local, required for Task 4)

Task 4 (AI commentary and LLM-driven QA) uses **Ollama only**, run locally. No API key is needed.

1. **Install Ollama**: [ollama.com](https://ollama.com/) (macOS/Windows/Linux).
2. **Pull and run a model** (Ollama serves at `http://localhost:11434` by default):

   ```bash
   ollama pull llama3.2
   ollama run llama3.2
   ```

3. **Run the pipeline** (default model is `llama3.2`; override with `export LLM_MODEL=mistral` if needed):

   ```bash
   python run_all.py
   ```

## Documentation

- **SUBMISSION.md** — Submission document (1–3 pages): requirement mapping and repo index. **Add your full name and email at the top** before submitting.
- **PIPELINE_GUIDE.md** — Guide to data, features, models, and trading interpretation.
- **PIPELINE_GUIDE.pdf** — Same as PDF (generate with `pandoc PIPELINE_GUIDE.md -o PIPELINE_GUIDE.pdf --pdf-engine=xelatex`).

## Outputs

- `data/` – hourly dataset + forecast predictions
- `reports/` – QA report, forecast metrics, curve sheet, AI log
- `figures/` – 3 figures (data quality, forecast vs actual, model comparison)
- `submission.csv` – out-of-sample predictions with columns **id** (timestamp UTC, hourly), **y_pred**. The first line is a comment stating the **test window** (e.g. `2025-10-01 to 2025-12-31`); the window is defined in `src/config.py` as `TEST_START` and `TEST_END`.

## Tasks

1. **Ingestion + QA**: Build hourly dataset; check missing, duplicates, outliers, coverage, DST.
2. **Forecasting**: Next-day hourly DA prices. Baseline: Seasonal Naive. Improved: Ridge, LightGBM. Walk-forward; MAE, RMSE, bias, and tail MAE (bottom/top 5% price hours). Spikes are the hardest regime; we quantify tail error explicitly.
3. **Curve translation**: Delivery means, uncertainty bands, BUY/SELL signal. Short note on desk usage and when to ignore the signal.
4. **AI**: Ollama (local) for daily commentary and LLM-driven QA; prompt and output logged.
