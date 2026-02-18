# Skill Extraction & Taxonomy Mapping Pipeline

A high-performance NLP pipeline for extracting professional skills from job advertisements and mapping them to the standard **ESCO (European Skills, Competences, Qualifications and Occupations)** taxonomy using semantic vector search.

## 🚀 System Design

The system implements a multi-stage pipeline optimized for semantic alignment.

### 1. Data Ingestion & Preprocessing
- **Source**: Raw job advertisements in JSONL format.
- **Cleaning**: HTML sanitization and whitespace normalization using `BeautifulSoup`.

### 2. LLM-Guided Skill Extraction
- **Engine**: Local LLM (Ollama) via the `instructor` library.
- **Mechanism**: Structured information extraction using Pydantic schemas. 
- **Strategy**: Extracts skill/requirement phrases (2-5 words) with surrounding context to maintain semantic integrity.

### 3. Taxonomy Indexing (Vector Database)
- **Database**: `ChromaDB` (Persistent).
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence-Transformers) for dense vector representation.
- **Indexing**: Full-text indexing of ~14k unique ESCO skill labels with HNSW-based cosine similarity search.

### 4. Semantic Mapping
- **Retrieval**: Top-K vector search against the ChromaDB index.
- **Filtering**: Dynamic similarity thresholding to maintain high precision.
- **Alignment**: Candidate retrieval with confidence scores (1 - Cosine Distance).

### 5. Automated Evaluation & Calibration
- **Ground Truth**: Secondary LLM-as-a-Judge (Llama 3.1) to verify semantic matches.
- **Metrics**: 
  - **Precision@1 / Precision@3**: Quantifies accuracy of the top-ranked taxonomy candidates.
  - **Skills Coverage**: Percentage of extracted skills successfully mapped to at least one verified taxonomy label.
  - **Job Coverage**: Percentage of job advertisements with at least one verified skill mapping.
  - **Brier Score**: Measures the calibration of vector similarity scores against ground-truth validation.

## 📊 Outputs

Upon completion, the pipeline generates the following files in the `data/` directory:

| File | Description |
| :--- | :--- |
| `final_results.json` | Comprehensive report containing global metrics, configuration parameters, and the final list of mapped skills for each job. |
| `pipeline_raw.json` | Detailed raw data of all skill pairings and their corresponding LLM judgments (used for debugging and analysis). |

## 🛠 Tech Stack

| Component | Tool / Library |
| :--- | :--- |
| **Orchestration** | Python 3.10+ |
| **LLM Interface** | Ollama, Instructor, Pydantic |
| **Vector Search** | ChromaDB |
| **Embeddings** | HuggingFace Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Metrics** | NumPy |
| **UI/UX** | tqdm (Progress tracking) |

## 🏃 Quick Start

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```
2. **LLM Server**: Ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.1:8b`).
3. **Run Pipeline**:
   ```bash
   python main.py
   ```

### 📋 Data Preparation
Before running the pipeline, ensure your input data is ready:
1. Place your job advertisements file (in JSONL format) into the `data/` directory.
2. Rename the file to `jobs.json`.
3. Ensure the mandatory `data/taxomony.csv` file is also present (this is the ESCO skills list).

---

