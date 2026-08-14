"""Configuration centralisée du projet RAG SportSee."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")

INPUT_DIR = os.getenv("INPUT_DIR", "inputs")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vector_db")
FAISS_INDEX_FILE = f"{VECTOR_DB_DIR}/faiss_index.idx"
DOCUMENT_CHUNKS_FILE = f"{VECTOR_DB_DIR}/document_chunks.pkl"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
SEARCH_K = int(os.getenv("SEARCH_K", "5"))

# Paramètres d'interface utilisés par MistralChat.py.
APP_TITLE = os.getenv("APP_TITLE", "NBA Analyst AI")
NAME = os.getenv("NAME", "NBA")


def require_mistral_api_key() -> str:
    """Retourne la clé Mistral ou arrête l'exécution avec un message explicite."""
    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY est absente ou vide. Créez le fichier '.env' à la "
            "racine du projet et ajoutez : MISTRAL_API_KEY=votre_cle_api"
        )
    return MISTRAL_API_KEY
