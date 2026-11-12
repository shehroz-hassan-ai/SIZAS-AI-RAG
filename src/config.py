import os
from dotenv import load_dotenv

load_dotenv()

# API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pinecone settings
INDEX_NAME = "rag-pdf-index"
EMBEDDING_DIM = 384
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

# Chunking defaults
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval defaults
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# Upload limit
MAX_FILE_SIZE_MB = 20

NOT_FOUND_MESSAGE = "The answer is not available in the provided document."


def validate_config():
    """Keys missing ho to saaf error de."""
    missing = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise ValueError(f"Missing in .env: {', '.join(missing)}")