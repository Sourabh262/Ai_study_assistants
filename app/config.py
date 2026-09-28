import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# LLM configuration
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# Embedding configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# RAG configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.05"))


# Application directories
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"


# Supported document types
SUPPORTED_FILE_TYPES = ["txt", "pdf"]


def validate_config() -> None:
    """Validate required application configuration."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Please add it to your .env file."
        )