"""Configuration centralisee du projet RAG SportSee."""

import os
import ssl
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

# Parametres d'interface utilises par MistralChat.py.
APP_TITLE = os.getenv("APP_TITLE", "NBA Analyst AI")
NAME = os.getenv("NAME", "NBA")

# Configuration SSL : positionner DISABLE_SSL_VERIFICATION=1 dans .env pour
# desactiver la verification SSL (utile en entreprise avec proxy/certificat
# auto-signe). Laisser vide ou a 0 sur un reseau personnel.
DISABLE_SSL_VERIFICATION = os.getenv("DISABLE_SSL_VERIFICATION", "0") == "1"


def configure_ssl() -> None:
    """Configure la verification SSL selon DISABLE_SSL_VERIFICATION dans .env.

    Sur un reseau d'entreprise avec proxy interceptant le HTTPS, positionnez
    DISABLE_SSL_VERIFICATION=1 dans .env pour desactiver la verification SSL.

    Sur un reseau personnel, laissez DISABLE_SSL_VERIFICATION=0 (ou vide) pour
    conserver la verification SSL normale.

    Attention : desactiver la verification SSL reduit la securite. A n'utiliser
    que dans un environnement de developpement controle.
    """
    if DISABLE_SSL_VERIFICATION:
        os.environ["SSL_CERT_FILE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        os.environ["CURL_CA_BUNDLE"] = ""
        ssl._create_default_https_context = ssl._create_unverified_context


def require_mistral_api_key() -> str:
    """Retourne la cle Mistral ou arrete l'execution avec un message explicite."""
    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY est absente ou vide. Creez le fichier '.env' a la "
            "racine du projet et ajoutez : MISTRAL_API_KEY=votre_cle_api"
        )
    return MISTRAL_API_KEY