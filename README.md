# Skill Extraction & Taxonomy Mapping Pipeline

An NLP pipeline that extracts professional skills from job advertisements and maps them to the **ESCO (European Skills, Competences, Qualifications and Occupations)** taxonomy using LLM-based extraction, semantic vector search, and LLM-as-a-Judge verification.

---

## Pipeline Overview

```
jobs.json (JSONL)
    │
    ▼
[1] HTML Cleaning          BeautifulSoup — strips tags, normalises whitespace
    │
    ▼
[2] Skill Extraction       Ollama llama3.1:8b + instructor → ["Python", "team leadership", ...]
    │
    ▼
[3] Vector Search          ChromaDB (all-MiniLM-L6-v2) → top-3 ESCO candidates per skill
    │
    ▼
[4] Threshold Filter       similarity >= 0.5 — removes low-confidence candidates
    │
    ▼
[5] LLM Judge              Ollama llama3.1:8b → [True, False, True] per candidate
    │
    ▼
[6] Evaluation             Precision@1/3, Coverage, Brier Score
    │
    ▼
final_results.json + console report
```

---

## Project Structure

```
├── main.py                  # Pipeline entry point
├── requirements.txt
├── data/
│   ├── jobs.json            # Input: job ads (JSONL format)
│   ├── taxomony.csv         # Input: ESCO taxonomy skills
│   ├── chroma_db/           # Auto-generated: vector index (cached)
│   ├── final_results.json   # Output: config + metrics + per-job results
│   └── results_without_evaluation.json  # Output: raw pairings before metrics
└── src/
    ├── config.py            # All constants and paths
    ├── extraction.py        # LLM skill extraction
    ├── taxonomy_index.py    # ChromaDB index build/load/search
    ├── mapping.py           # Vector search + threshold filtering
    ├── judge.py             # LLM-as-a-Judge verification
    ├── evaluation.py        # Metrics computation and reporting
    └── exploration.py       # EDA on jobs and taxonomy data
```

---

## Metrics

| Metric | Description |
|---|---|
| **Precision@1** | % of extracted skills where the top-ranked candidate is verified correct |
| **Precision@3** | Average fraction of correct candidates across top-3 results |
| **Skills Coverage** | % of extracted skills with at least 1 verified taxonomy match |
| **Job Coverage** | % of job ads with at least 1 verified skill mapping |
| **Brier Score** | Calibration of similarity scores vs judge outcomes (lower = better) |
| **Confidence (median)** | Median similarity score of the first verified match |

> **Expected**: Precision@1 >= Precision@3 — vector search ranks the best candidate first, so positions 2 and 3 are naturally less accurate.

---

## Configuration

All settings are in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.1:8b` | Ollama model for extraction and judging |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model for vector search |
| `SIMILARITY_THRESHOLD` | `0.5` | Minimum cosine similarity to accept a candidate |
| `TOP_K` | `3` | Number of taxonomy candidates retrieved per skill |
| `SAMPLE_SIZE` | `2` | Number of job ads to process per run |

---

## Tech Stack

| Component | Library |
|---|---|
| LLM inference | Ollama + `openai` compatible client |
| Structured LLM output | `instructor` + `pydantic` |
| Vector database | `chromadb` |
| Embeddings | `sentence-transformers` |
| HTML parsing | `beautifulsoup4` |
| Metrics | `numpy` |
| Progress display | `tqdm` |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull the model
```bash
ollama serve
ollama pull llama3.1:8b
```

### 3. Prepare data
Place input files in `data/`:
- `data/jobs.json` — job advertisements in JSONL format (one JSON object per line)
- `data/taxomony.csv` — ESCO taxonomy CSV (included)

### 4. Run the pipeline
```bash
python main.py
```

### 5. (Optional) Run data exploration
```bash
# From workspace root:
python -m src.exploration

> The ChromaDB index is built on first run and cached in `data/chroma_db/`. Subsequent runs load it from disk — no re-embedding needed.

---

## Output Files

| File | Contents |
|---|---|
| `data/final_results.json` | Config + all metrics + per-job skill mappings |
| `data/results_without_evaluation.json` | Raw pairings + judgments before metric computation (useful for re-running evaluation without re-calling the LLM) |
