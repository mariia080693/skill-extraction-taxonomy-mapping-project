"""
  Skill Extraction & Taxonomy Mapping Pipeline:
  1) LLM-based extraction (Ollama)
  2) Sentence-embedding vector search against ESCO taxonomy skills
"""

import json
import time

from transformers import logging
from tqdm import tqdm

from src.config import (
    JOBS_PATH,
    RESULTS_PATH,
    SAMPLE_SIZE,
    SIMILARITY_THRESHOLD,
    RESULTS_NO_EVALUATION_PATH,
)
from src.evaluation import full_report, print_report, print_sample_mappings
from src.extraction import extract_skills
from src.mapping import map_all_skills
from src.taxonomy_index import TaxonomyIndex

logging.set_verbosity_error() # suppress transformers warnings

# Load job advertisements from a JSONL file.
def load_jobs(path: str = JOBS_PATH, limit: int | None = SAMPLE_SIZE) -> list[dict]:
    jobs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(jobs) >= limit:
                break
            jobs.append(json.loads(line))
    return jobs

# Extract skill phrases from a job and map them to taxonomy skills.
def process_job(job: dict, index: TaxonomyIndex) -> dict:
    
    # 1) Extraction
    skills = extract_skills(job)

    # 2) Taxonomy mapping
    # pairing is a List[Dict], where each Dict has 'extracted_job_skill' and 'candidates' (List[Dict])
    pairing = map_all_skills(skills, index)
    
    return {
        "job_id": job.get("id"),
        "title": job.get("title"),
        "pairing": pairing, 
    }

# Main pipeline.
from src.judge import judge_mappings

def main():
    print("\nSKILL EXTRACTION & TAXONOMY MAPPING PIPELINE")
  
    # 1. Load data
    jobs = load_jobs(limit=SAMPLE_SIZE)
    print(f"\n[1/5] Loaded {len(jobs)} job ads")

    # 2. Prepare taxonomy index
    print("\n[2/5] Preparing taxonomy index")
    index = TaxonomyIndex.get_or_build()
    print(f"Index ready: {index.total_skills:,} skill vectors")

    # 3. Process jobs
    print("\n[3/5] Extracting skills & mapping to taxonomy")
    results = []     # List of Jobs (metadata + all skill pairings) 
    
    start_time = time.time()
    
    for job in tqdm(jobs, desc="Processing"):
        try:
            result = process_job(job, index)
            results.append(result)
        except Exception as e:
            print(f"Skipped job {job.get('id')}: {e}")

    elapsed_time = time.time() - start_time
    print(f"Processing complete in {elapsed_time:.3f}s")

    # 4. LLM Judging
    print("\n[4/5] LLM Judging (Ground Truth Verification)")
    # Iterate through all skill pairings and add judgment
    
    start_time = time.time()
    
    all_pairing = [p for job in results for p in job.get("pairing", [])]
    for p in tqdm(all_pairing, desc="Judging"):
        if p.get("candidates"):
             # p is a dict ref
             p["judgment"] = judge_mappings(p["extracted_job_skill"], p["candidates"])

    elapsed_time = time.time() - start_time
    
    print(f"Judging complete in {elapsed_time:.3f}s")
    
    # 5. Evaluate & Report
    print("\n[5/5] Evaluation & Reporting\n")
    
    # Save raw data for reuse
    raw_output = {
        "results_before_evaluation": results,
    }
    
    with open(RESULTS_NO_EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_output, f, indent=2, ensure_ascii=False)
    print(f"Results without evaluation saved → {RESULTS_NO_EVALUATION_PATH}")

    report = full_report(results, threshold=SIMILARITY_THRESHOLD)
    
    # Save results
    output = {
        "config": {
            "sample_size": len(results),
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
        "metrics": report,
        "jobs": results,  
    }
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary to console
    print_report(report, len(results), elapsed_time)
    print_sample_mappings(results)
    
    print(f"\nResults saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
