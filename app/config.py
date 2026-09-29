import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


def get_env_variable(var_name: str, default: str | None = None) -> str | None:
    """Retrieve an environment variable from os.environ or Streamlit secrets."""
    val = os.getenv(var_name)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and var_name in st.secrets:
            return str(st.secrets[var_name])
    except Exception:
        pass
    return default


# Environment
GROQ_API_KEY = get_env_variable("GROQ_API_KEY")


# LLM configuration
GROQ_MODEL = get_env_variable(
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