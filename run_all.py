# Run full pipeline: ingest → baseline QA → dataset + QA narrative → forecast → curve → commentary → submission
import time
from src import main as t1
from src import forecast as t2
from src import curve_translation as t3
from src import ai_commentary as t4
from src import generate_submission as gen

if __name__ == "__main__":
    t0 = time.time()
    print("Pipeline: DE-LU")
    t1.main()
    t4.run_llm_qa_and_save_cleaned()
    t2.main()
    t3.main()
    t4.main()
    gen.main()
    print(f"Done in {time.time()-t0:.0f}s")
