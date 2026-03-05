# Paths
JOBS_PATH = "data/jobs.json"
TAXONOMY_PATH = "data/taxomony.csv"
CHROMA_DIR = "data/chroma_db"          
CHROMA_COLLECTION = "taxonomy"    # ChromaDB collection name
RESULTS_PATH = "data/final_results.json"
RESULTS_NO_EVALUATION_PATH = "data/results_without_evaluation.json"

# LLM (Ollama) to avoid API costs
LLM_BASE_URL = "http://localhost:11434/v1" 
LLM_MODEL = "llama3.1:8b"
LLM_TEMPERATURE = 0.0

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"        

# Pipeline
SAMPLE_SIZE = 2 # number of jobs to process
SIMILARITY_THRESHOLD = 0.5
TOP_K = 3 # number of taxonomy candidates to return for each extracted skill

