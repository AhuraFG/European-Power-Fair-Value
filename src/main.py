# Task 1: Ingest + baseline QA → save baseline parquet
from pathlib import Path
from . import config
from . import ingest
from . import qa_checks

def main():
    print("Task 1: Data ingestion + baseline QA")
    df = ingest.run_ingestion()
    print(f"  Shape {df.shape}")
    qa_checks.run_qa(df)
    df = df[~df.index.duplicated(keep="first")].dropna(subset=["price_eur_mwh"])
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    df.to_parquet(Path(config.DATA_DIR) / config.BASELINE_PARQUET)
    print(f"  Baseline → {config.DATA_DIR}/{config.BASELINE_PARQUET} ({len(df)} rows)")
    print("Done.")

if __name__ == "__main__":
    main()
