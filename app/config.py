"""Environment and constants for the Emporio da Musica support agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

DATA_DIR = REPO_ROOT / "data"
STORE_DB_PATH = Path(os.getenv("STORE_DB_PATH", REPO_ROOT / "store.db"))
POLICY_INDEX_DIR = Path(os.getenv("POLICY_INDEX_DIR", REPO_ROOT / "index"))

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "8"))

ASSISTANT_NAME = "Empório da Música"
STORE_SLOGAN = "Sua música começa aqui."

# Every CSV in the challenge shares this prefix.
CSV_PREFIX = "desafio_tecnico_ai_eng - "
POLICY_PDF = DATA_DIR / "politicas_da_loja.pdf"
