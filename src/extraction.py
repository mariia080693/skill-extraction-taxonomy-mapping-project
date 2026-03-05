"""
Extract skill/requirement data from raw job ads using HTML cleaning and a local LLM.
"""

import re
from typing import List

import instructor
from bs4 import BeautifulSoup
from openai import OpenAI # OpenAI-compatible client for Ollama
from pydantic import BaseModel, Field

from src.config import LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

# Pydantic schema for LLM output
class ExtractedSkills(BaseModel):
    skills: List[str] = Field(default_factory=list,
                              description="all extracted job ad skills, responsibilities, and requirements")


#  HTML tags cleaner with BeautifulSoup
def clean_html(raw_html: str) -> str:
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ") # extracts text from the HTML parse tree, removes all tags, and joins with spaces
    return re.sub(r"\s+", " ", text).strip() # normalize whitespace


#  LLM client

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = instructor.from_openai(
            OpenAI(base_url=LLM_BASE_URL, api_key="ollama"),
            mode=instructor.Mode.JSON,
        )
    return _client


# Extraction prompt 

SYSTEM_PROMPT = (
    "You are a precise skills, responsibilities, and requirements information-extraction assistant. "
    "Given a job advertisement, extract all skills, responsibilities, and requirements "
    "into a single list called 'skills'. "
    "Return skill/requirement phrases (2-5 words) with some context (up to 10 words each) in the same list."
    "Example: if the job ad says 'Driving buses to school every Wednesday in the morning is required', you might extract 'Driving' for a skill phrase and 'buses to school' for context, and return 'Driving buses to school' in the skills list. "
    "If no skills are found, return an empty list."
)
# extract title, abstract and main content('content' + 'metadata.standout') from a job ad
def extract_skills(job: dict) -> List[str]:
    title = job.get("title", "")
    abstract = job.get("abstract", "")
    content = clean_html(job.get("content", "") + " " + str(job.get("metadata", {}).get("standout", "")))

    # Concatenation of the job ad info into a single string to send to the LLM for extraction
    body = f"Title: {title}\nAbstract: {abstract}\nMain job description: {content}"

    client = _get_client() # get or create LLM client
    result = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        response_model=ExtractedSkills, # format the LLM response
        max_retries=2, 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": body}, # jod ad info is a user message
        ],
    )
    return result.skills # returns the 'skills' field defined in the ExtractedSkills
