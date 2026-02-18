"""
LLM-as-a-Judge module to evaluate skill mappings.
"""

from typing import List, Dict
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from src.config import LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE


_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = instructor.from_openai(
            OpenAI(base_url=LLM_BASE_URL, api_key="ollama"),
            mode=instructor.Mode.JSON,
        )
    return _client

class Judgment(BaseModel):
    matches: List[bool] = Field(
        ..., 
        description="List of booleans interacting with the provided candidates order. True if the candidate is a semantic match for the extracted skill, False otherwise."
    )

JUDGE_SYSTEM_PROMPT = (
    "You are an expert taxonomist and evaluator. "
    "Your task is to judge whether a set of candidate standardized skills are valid semantic matches for an extracted job skill. "
    "Input: 'Extracted Skill' and a list of 'Candidates'. "
    "Output: A list of boolean values (true/false) corresponding strictly to the order of candidates. "
    "Return true if the candidate represents the SAME core skill or a very close synonym/variant. "
    "Return false if it is unrelated, too broad, too narrow, or just shares a keyword without sharing meaning."
)
# Ask the LLM to judge a list of candidates for a given extracted skill.
def judge_mappings(extracted_skill: str, candidates: List[Dict]) -> List[bool]:
    
    if not candidates:
        return []

    # Format the input for the LLM
    candidate_text = "\n".join([f"{i+1}. {c['label']}" for i, c in enumerate(candidates)])
    user_content = f"Extracted Skill: '{extracted_skill}'\n\nCandidates:\n{candidate_text}"

    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0, # Deterministic
            response_model=Judgment,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_retries=2,
        )
        
        # Ensure we return valid length
        results = resp.matches
        # Pad or truncate if the LLM messes up length
        if len(results) < len(candidates):
            results.extend([False] * (len(candidates) - len(results)))
        return results[:len(candidates)]
    except Exception as e:
        print(f"Judge error for '{extracted_skill}': {e}")
        return [False] * len(candidates)
