from __future__ import annotations

import os
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


BASE_DIR = Path(__file__).resolve().parent

APP_NAME = os.getenv("APP_NAME", "Diabetes Care Chat")
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
DEBUG = _bool_env("DEBUG", APP_ENV != "production")

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "saved_indexes"))
FAISS_DIR = Path(os.getenv("FAISS_DIR", INDEX_DIR / "faiss"))
BM25_PATH = Path(os.getenv("BM25_PATH", INDEX_DIR / "bm25.pkl"))
INDEX_MANIFEST_PATH = Path(os.getenv("INDEX_MANIFEST_PATH", INDEX_DIR / "manifest.json"))

MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "gemma-diabetes-q8_0.gguf"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "70"))

BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
FAISS_WEIGHT = float(os.getenv("FAISS_WEIGHT", "0.6"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-10"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "600"))

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
MAX_FEEDBACK_CHARS = int(os.getenv("MAX_FEEDBACK_CHARS", "12000"))
MAX_SUGGESTION_CHARS = int(os.getenv("MAX_SUGGESTION_CHARS", "2000"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "7200"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))

FEEDBACK_DB = Path(os.getenv("FEEDBACK_DB", BASE_DIR / "feedback.sqlite3"))

RATE_LIMIT_ENABLED = _bool_env("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

SECURITY_HEADERS_ENABLED = _bool_env("SECURITY_HEADERS_ENABLED", True)
ALLOWED_HOSTS = _csv_env("ALLOWED_HOSTS", os.getenv("ALLOWED_HOST", "*")) or ["*"]
CORS_ORIGINS = _csv_env("CORS_ORIGINS", "")

ALLOW_DANGEROUS_INDEX_DESERIALIZATION = _bool_env("ALLOW_DANGEROUS_INDEX_DESERIALIZATION", True)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_GUARDRAIL_ID = os.getenv("BEDROCK_GUARDRAIL_ID", "")
BEDROCK_GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
BEDROCK_MODERATION_ENABLED = _bool_env("BEDROCK_MODERATION_ENABLED", False)
BEDROCK_MODERATION_TIMEOUT_SECONDS = float(os.getenv("BEDROCK_MODERATION_TIMEOUT_SECONDS", "3.0"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)

