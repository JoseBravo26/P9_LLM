"""Interface Streamlit de l assistant NBA Analyst AI.

Cette version branche le routeur SQL/RAG (utils/rag_pipeline_router.py) :
    - les questions chiffrees/comparatives sont traitees par le SQL Tool sur
      la base SQLite (players, matches, stats, reports) ;
    - les questions narratives restent traitees par le pipeline RAG FAISS
      existant sur les rapports PDF.
Le routage, le traçage Logfire et la gestion d erreurs sont conserves de la
version precedente ; seule la logique de reponse est deleguee au routeur.
"""

import logging

import streamlit as st
from dotenv import load_dotenv

from utils.config import APP_TITLE, MODEL_NAME, NAME
from utils.rag_pipeline_router import answer
from utils.vector_store import VectorStoreManager

load_dotenv()

try:
    import logfire
    logfire.configure()
    logfire.info("Logfire configure pour NBA Analyst AI")
except ImportError:
    logfire = None
    logging.warning("Logfire non installe : desactivation du tracage")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)

ROUTE_LABELS = {
    "SQL": "📊 Statistiques (base SQL)",
    "RAG": "📄 Analyse documentaire (rapports PDF)",
}


@st.cache_resource
def get_vector_store_manager() -> VectorStoreManager | None:
    """Charge l index vectoriel existant une seule fois par session Streamlit."""
    try:
        manager = VectorStoreManager()

        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel est absent ou vide.")
            st.warning(
                "Executez 'python indexer.py' apres avoir ajoute les fichiers dans 'inputs'."
            )
            return None

        logging.info("Index charge : %s vecteurs.", manager.index.ntotal)
        if logfire:
            logfire.info("Index FAISS charge", n_vectors=manager.index.ntotal)
        return manager
    except (FileNotFoundError, RuntimeError) as error:
        st.error(str(error))
        return None
    except Exception as error:
        logging.exception("Erreur lors du chargement du Vector Store")
        st.error(f"Erreur inattendue lors du chargement de l'index : {error}")
        return None


def afficher_reponse(question: str, store: VectorStoreManager) -> tuple[str, str]:
    """Route la question, affiche le badge de source et retourne (route, texte de reponse)."""
    if logfire:
        with logfire.span("agent.answer", question=question[:200]):
            result = answer(question, store=store)
    else:
        result = answer(question, store=store)

    route = result["route"]
    response = result["response"]

    st.caption(ROUTE_LABELS.get(route, route))

    if response.abstained:
        st.info(response.answer)
    else:
        st.write(response.answer)
        if response.cited_chunk_ids:
            with st.expander("Sources citees"):
                st.write(", ".join(response.cited_chunk_ids))

    if logfire:
        logfire.info(
            "Reponse affichee",
            route=route,
            abstained=response.abstained,
            confidence=response.confidence,
        )

    return route, response.answer


def main() -> None:
    """Execute l application Streamlit."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🏀")
    st.title(APP_TITLE)
    st.caption(f"Assistant virtuel pour {NAME} | Modele : {MODEL_NAME}")
    st.caption(
        "Questions chiffrees -> base SQL (players/stats). "
        "Questions d'analyse -> rapports PDF (RAG)."
    )

    vector_store_manager = get_vector_store_manager()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"Bonjour ! Je suis votre analyste IA pour la {NAME}. "
                    "Posez-moi une question chiffree (statistiques) ou une question "
                    "d'analyse sur les rapports de match."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input(f"Posez votre question sur la {NAME}...")

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    if vector_store_manager is None:
        st.warning(
            "L'index documentaire est indisponible : les questions d'analyse PDF "
            "ne pourront pas etre traitees. Les questions chiffrees restent possibles."
        )

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            try:
                _, response_content = afficher_reponse(prompt, vector_store_manager)
            except Exception as error:
                logging.exception("Erreur lors du traitement de la question")
                if logfire:
                    logfire.error("Erreur pipeline agent", error=str(error))
                response_content = (
                    "Une erreur technique empeche la generation de la reponse : "
                    f"{error}"
                )
                st.error(response_content)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_content}
    )


if __name__ == "__main__":
    main()
