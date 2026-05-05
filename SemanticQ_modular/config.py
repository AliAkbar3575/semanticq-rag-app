import os
from dotenv import load_dotenv

load_dotenv()

# model configurations
GROQ_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# vector database configurations
VECTOR_DB_PATH = "models/faiss_index"
SEARCH_K = 3

# data configurations
DATA_PATH = "data/faqs.csv"

# logging configurations
LOG_FILE = "logs/semanticq.log"
LOG_LEVEL = "INFO"

# API configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")