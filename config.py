import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API Key configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Embedding Model Configuration (Semantic Representation)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Text Chunking Hyperparameters
DEFAULT_CHUNK_SIZE = 300      # Number of words per chunk
DEFAULT_CHUNK_OVERLAP = 50    # Word overlap to maintain context between chunks

# Search and Hybrid Fusion (RRF) Hyperparameters
DEFAULT_TOP_K = 3             # Number of top documents to retrieve
RRF_K = 60                    # Reciprocal Rank Fusion constant

# Minimum similarity score threshold to ensure the retrieved chunk is relevant to the user's query.
MIN_SIMILARITY_SCORE = 0.25