"""Script de construction de l'index vectoriel FAISS."""

import argparse
import logging
from typing import Optional

from utils.config import INPUT_DIR, require_mistral_api_key
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_indexing(input_directory: str, data_url: Optional[str] = None) -> None:
    """Exécute le chargement, le découpage et l'indexation des documents."""
    require_mistral_api_key()
    logging.info("--- Démarrage du processus d'indexation ---")

    if data_url:
        logging.info("Tentative de téléchargement depuis l'URL configurée.")
        success = download_and_extract_zip(data_url, input_directory)

        if not success:
            logging.error("Échec du téléchargement ou de l'extraction. Arrêt.")
            return
    else:
        logging.info(
            "Aucune URL fournie. Utilisation des fichiers locaux dans : %s",
            input_directory,
        )

    logging.info("Chargement et parsing des fichiers depuis : %s", input_directory)
    documents = load_and_parse_files(input_directory)

    if not documents:
        logging.warning(
            "Aucun document n'a été chargé. Vérifiez le dossier d'entrée."
        )
        logging.info("--- Processus terminé : aucun document traité ---")
        return

    logging.info("Initialisation du gestionnaire de Vector Store.")
    vector_store = VectorStoreManager()

    logging.info("Construction de l'index FAISS : cette étape peut prendre du temps.")
    vector_store.build_index(documents)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info("Nombre de documents traités : %s", len(documents))

    if vector_store.index is not None:
        logging.info("Nombre de chunks indexés : %s", vector_store.index.ntotal)
    else:
        logging.warning("L'index final est absent ou vide.")


def parse_arguments() -> argparse.Namespace:
    """Définit et lit les arguments du script d'indexation."""
    parser = argparse.ArgumentParser(
        description="Script d'indexation pour l'application RAG"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire des fichiers sources (défaut : {INPUT_DIR})",
    )
    parser.add_argument(
        "--data-url",
        type=str,
        default=None,
        help="URL optionnelle vers un fichier ZIP contenant les sources.",
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
        logging.error("Indexation interrompue : %s", error)
        raise SystemExit(1) from error
