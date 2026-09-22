"""Script de construction de l'index vectoriel FAISS à partir des documents narratifs.

Les fichiers structurés, notamment Excel, sont volontairement exclus du RAG :
ils sont traités par le pipeline SQL (`load_excel_to_db.py`) et introduiraient
du bruit, des valeurs NaN et des en-têtes de tableaux dans les chunks FAISS.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from utils.config import INPUT_DIR, configure_ssl, require_mistral_api_key
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.vector_store import VectorStoreManager

configure_ssl()
require_mistral_api_key()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

RAG_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def filter_rag_documents(documents: list[dict]) -> list[dict]:
    """Conserve uniquement les documents textuels adaptés à la recherche RAG.

    Les tableurs et sources structurées doivent être interrogés avec le SQL Tool.
    Cette séparation évite que les lignes vides, les valeurs NaN et les en-têtes
    Excel dégradent la qualité de récupération dans FAISS.
    """
    filtered_documents = []

    for document in documents:
        metadata = document.get("metadata", {})
        filename = metadata.get("filename", "")
        extension = Path(filename).suffix.lower()

        if extension in RAG_SUPPORTED_EXTENSIONS:
            filtered_documents.append(document)
        else:
            LOGGER.info(
                "Document exclu de l'index RAG : %s (extension %s).",
                filename,
                extension or "inconnue",
            )

    return filtered_documents


def run_indexing(input_directory: str, data_url: Optional[str] = None) -> None:
    """Charge, filtre et indexe les documents narratifs pour le RAG."""
    LOGGER.info("--- Démarrage du processus d'indexation RAG ---")

    if data_url:
        LOGGER.info("Tentative de téléchargement depuis l'URL configurée.")
        if not download_and_extract_zip(data_url, input_directory):
            LOGGER.error("Échec du téléchargement ou de l'extraction. Arrêt.")
            return
    else:
        LOGGER.info(
            "Aucune URL fournie. Utilisation des fichiers locaux dans : %s",
            input_directory,
        )

    all_documents = load_and_parse_files(input_directory)
    if not all_documents:
        LOGGER.warning("Aucun document n'a été chargé.")
        return

    documents = filter_rag_documents(all_documents)
    if not documents:
        LOGGER.warning(
            "Aucun document textuel compatible RAG n'a été trouvé. "
            "Ajoutez des PDF, DOCX, TXT ou Markdown dans le dossier d'entrée."
        )
        return

    LOGGER.info(
        "Documents retenus pour le RAG : %s sur %s.",
        len(documents),
        len(all_documents),
    )

    vector_store = VectorStoreManager()
    vector_store.build_index(documents)

    if vector_store.index is None:
        raise RuntimeError("L'index FAISS final est absent ou vide.")

    LOGGER.info("--- Indexation RAG terminée avec succès ---")
    LOGGER.info("Documents narratifs indexés : %s", len(documents))
    LOGGER.info("Chunks indexés : %s", vector_store.index.ntotal)


def parse_arguments() -> argparse.Namespace:
    """Définit et lit les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Indexation des documents narratifs pour le RAG."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire contenant les documents sources (défaut : {INPUT_DIR}).",
    )
    parser.add_argument(
        "--data-url",
        type=str,
        default=None,
        help="URL facultative d'une archive ZIP de documents.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    try:
        run_indexing(
            input_directory=arguments.input_dir,
            data_url=arguments.data_url,
        )
    except RuntimeError as error:
        LOGGER.error("Indexation interrompue : %s", error)
        raise SystemExit(1) from error