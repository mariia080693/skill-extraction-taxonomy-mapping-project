
"""
Core metric  : Precision@1 (with similarity threshold)
"""

from typing import Dict, List
import numpy as np



# Check if the mapping has at least one candidate confirmed by the judge
def has_valid_match(mapping: Dict) -> bool:
    return "judgment" in mapping and any(mapping["judgment"])

#  Precision@1
def precision_at_1(mappings: List[Dict]) -> float:
    if not mappings:
        return 0.0
    # A mapping is considered a "hit" if the JUDGE says True to the FIRST candidate
    hits = 0
    for m in mappings:
        if "judgment" in m and m["judgment"] and m["judgment"][0]:
            hits += 1
    return hits / len(mappings)

# Average Precision@3 (based on valid judgments)
def precision_at_3(mappings: List[Dict]) -> float:
    judged = [m["judgment"][:3] for m in mappings if "judgment" in m]
    if not judged:
        return 0.0
    return sum(sum(j) / 3.0 for j in judged) / len(judged)


def avg_candidates_per_skill(mappings: List[Dict]) -> float:
    if not mappings:
        return 0.0
    # Only count candidates that were judged as True
    total_valid = 0
    for m in mappings:
        if "judgment" in m:
            total_valid += sum(1 for is_match in m["judgment"] if is_match)
    return total_valid / len(mappings)


# Coverage (extracted skills - mapped skills)

def mapping_coverage(mappings: List[Dict]) -> Dict[str, float]:
    total = len(mappings)
    mapped = sum(1 for m in mappings if has_valid_match(m))
    return {
            "skills_total_extracted_job_skills": total,
            "skills_mapped": mapped,
            "skills_unmapped": total - mapped,
            "skills_coverage": round(mapped / total, 4) if total else 0.0,
        }


# Job coverage (percentage of jobs with at least 1 JAUDGED mapped skill)

def job_coverage(results: List[Dict]) -> Dict[str, float]:
    if not results:
        return {
            "total_jobs": 0,
            "jobs_with_mappings": 0,
            "job_coverage": 0.0,
        }
    
    total_jobs = len(results)
    jobs_with_mappings = sum(
        1 for job in results
        if job.get("pairing") and any(has_valid_match(m) for m in job["pairing"])
    )
    return {
        "total_jobs": total_jobs,
        "jobs_with_mappings": jobs_with_mappings,
        "job_coverage": round(jobs_with_mappings / total_jobs, 4) if total_jobs else 0.0,
    }


# Confidence distribution (median confidence of the BEST JUDGED match)

def confidence_stats(mappings: List[Dict]) -> Dict[str, float]:
    scores = []
    for m in mappings:
        if has_valid_match(m):
            # Find the score of the *first* candidate that is judged True
            for i, is_match in enumerate(m["judgment"]):
                if is_match and i < len(m["candidates"]):
                    scores.append(m["candidates"][i]["score"])
                    break
                    
    if not scores:
        return {"median": 0}
    arr = np.array(scores)
    return {
        "median": round(float(np.median(arr)), 4),
    }


# Brier calibration score (Mean Squared Error between confidence and judgment)
def brier_score(mappings: List[Dict]) -> float:

    squared_errors = []
    for m in mappings:
        candidates = m.get("candidates", [])
        judgments = m.get("judgment", [])
        
        for i, c in enumerate(candidates):
            # Outcome is 1.0 if judgment is True, else 0.0
            outcome = 1.0 if (i < len(judgments) and judgments[i]) else 0.0
            confidence = c["score"]
            squared_errors.append((confidence - outcome) ** 2)
            
    if not squared_errors:
        return 0.0
        
    return round(sum(squared_errors) / len(squared_errors), 4)


# A full evaluation report

def full_report(all_pairing: List[Dict], results: List[Dict], threshold: float) -> Dict:

    return {
        "precision@1": precision_at_1(all_pairing),
        "precision@3": precision_at_3(all_pairing),
        "avg_candidates": avg_candidates_per_skill(all_pairing),
        "skills_coverage": mapping_coverage(all_pairing),
        "job_coverage": job_coverage(results),
        "confidence": confidence_stats(all_pairing),
        "brier_score": brier_score(all_pairing),
    }


def print_report(report: Dict, results_count: int, elapsed_time: float = None):
    """
    Print a formatted summary of the evaluation report.
    """
    print("\nRESULTS SUMMARY\n")
    print(f"Jobs processed: {results_count:,}")
    if elapsed_time:
         print(f"Processing time: {elapsed_time/max(results_count, 1):.2f}s/job")

    #print(f"Jobs with >= 1 verified skill: {report['job_coverage']['jobs_with_mappings']:,}")
    print(f"Job coverage (>=1 verified skills): {report['job_coverage']['job_coverage']:.2%}")
    print(f"Total skills extracted: {report['skills_coverage']['skills_total_extracted_job_skills']:,}")
    print(f"Skills mappings: {report['skills_coverage']['skills_mapped']:,}")
    print(f"Skills unverified/no match: {report['skills_coverage']['skills_unmapped']:,}")
    print(f"Skills coverage: {report['skills_coverage']['skills_coverage']:.2%}")
    print(f"Precision@1: {report['precision@1']:.2f}")
    if "precision@3" in report:
        print(f"Precision@3: {report['precision@3']:.2f}")
    print(f"Avg verified candidates per skill: {report['avg_candidates']:.2f}")
    
    print(f"Calibration (Brier Score): {report['brier_score']:.4f}")
    print(f"Confidence (median of top verified match): {report['confidence']['median']:.2f}")


# Print a few example mappings from the results.
def print_sample_mappings(results: List[Dict], count: int = 3):
    """
    Print a few example mappings from the results using judgment keys.
    """
    print("\n\nSAMPLE MAPPINGS (Verified by LLM)\n")
    
    for r in results[:count]:
        print(f"\n Job: {r['title']} (id={r['job_id']})\n")
        
        for m in r["pairing"]:
            print(f'Original: "{m["extracted_job_skill"]}"')
            
            judgments = m.get("judgment", [])
            candidates = m.get("candidates", [])
            
            if not candidates:
                 print("  - (no candidates found)")
            else:
                for i, c in enumerate(candidates):
                    is_match = judgments[i] if i < len(judgments) else False
                    status = "[VERIFIED]" if is_match else "[REJECTED]"
                    print(f'  - {status} "{c["label"]}" [{c["score"]:.3f}]')
            
            print()

