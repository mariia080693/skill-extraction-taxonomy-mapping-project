"""
Map extracted skill phrases to the ESCO taxonomy via vector search.
"""

from typing import Dict, List
from src.config import SIMILARITY_THRESHOLD, TOP_K
from src.taxonomy_index import TaxonomyIndex

# Map a single data phrase to the best matching taxonomy skill (if > threshold)
def map_skill_to_taxonomy(
    skill_phrase: str,
    index: TaxonomyIndex,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Dict:
    
    # Search for more candidates initially to allow for filtering
    raw_candidates = index.search(skill_phrase, top_k=top_k) 
    
    # Filter by THRESHOLD
    valid_candidates = [
        {"label": label, "score": round(score, 3)}
        for label, score in raw_candidates
        if score >= threshold
    ]
    
    # Sort descending and take top_k
    valid_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "extracted_job_skill": skill_phrase,
        "candidates": valid_candidates, # List of dicts: {label, score}
    }

def map_all_skills(skill_phrases: List[str], index: TaxonomyIndex) -> List[Dict]:
    return [map_skill_to_taxonomy(s, index) for s in skill_phrases]
