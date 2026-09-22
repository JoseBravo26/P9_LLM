"""Interface Streamlit de NBA Analyst AI.

L'interface délègue la réponse au routeur SQL/RAG :
- les questions statistiques sont traitées par la base SQLite ;
- les questions narratives sont traitées par le pipeline RAG sur les PDF ;
- les réponses insuffisamment étayées déclenchent une abstention explicite.
"""

import logging
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from utils.config import APP_TITLE, MODEL_NAME, NAME, configure_ssl, require_mistral_api_key

configure_ssl()
load_dotenv()
require_mistral_api_key()

from utils.rag_pipeline_router import answer
from utils.vector_store import VectorStoreManager

try:
    import logfire
except ImportError:
    logfire = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

if logfire:
    logfire.configure()
    logfire.info("Logfire configuré pour NBA Analyst AI")

ROUTE_LABELS = {
    "SQL": "📊 Statistiques (base SQL)",
    "RAG": "📄 Analyse documentaire (rapports PDF)",
}


@st.cache_resource
def get_vector_store_manager() -> Optional[VectorStoreManager]:
    """Charge l'index vectoriel une seule fois par session Streamlit."""
    try:
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            LOGGER.warning("Index vectoriel absent ou vide.")
            return None

        LOGGER.info("Index chargé : %s vecteurs.", manager.index.ntotal)
        if logfire:
            logfire.info("Index FAISS chargé", n_vectors=manager.index.ntotal)
        return manager
    except (FileNotFoundError, RuntimeError) as error:
        LOGGER.exception("Chargement de l'index impossible : %s", error)
        return None
    except Exception as error:
        LOGGER.exception("Erreur inattendue lors du chargement du Vector Store")
        if logfire:
            logfire.error("Erreur chargement index", error=str(error))
        return None


def afficher_extraits_diagnostic(contexts: list) -> None:
    """Affiche les extraits récupérés lorsque le RAG s'abstient.

    Ces extraits ne sont pas présentés comme des citations de réponse : ils
    permettent d'expliquer pourquoi l'information disponible est insuffisante.
    """
    if not contexts:
        return

    with st.expander("Extraits récupérés pour diagnostic"):
        st.caption(
            "Ces passages ont été récupérés par la recherche vectorielle, mais "
            "ils ne suffisent pas à produire une réponse fiable."
        )
        for chunk in contexts:
            filename = getattr(chunk.metadata, "filename", "Source inconnue")
            st.markdown(
                f"**{filename} — chunk {chunk.id} "
                f"(similarité : {chunk.score:.3f})**"
            )
            st.write(chunk.text[:600])
            st.divider()


def afficher_reponse(question: str, store: VectorStoreManager) -> tuple[str, str]:
    """Route la question, affiche le résultat et retourne ``(route, réponse)``."""
    if logfire:
        with logfire.span("agent.answer", question=question[:200]):
            result = answer(question, store=store)
    else:
        result = answer(question, store=store)

    route = result["route"]
    response = result["response"]
    contexts = result.get("contexts", [])

    st.caption(ROUTE_LABELS.get(route, route))

    if response.abstained:
        st.info(response.answer)
        if route == "RAG":
            afficher_extraits_diagnostic(contexts)
    else:
        st.write(response.answer)
        if response.cited_chunk_ids:
            with st.expander("Sources citées"):
                st.write(", ".join(response.cited_chunk_ids))

    if logfire:
        logfire.info(
            "Réponse affichée",
            route=route,
            abstained=response.abstained,
            confidence=response.confidence,
            retrieved_chunks=len(contexts),
        )

    return route, response.answer


def main() -> None:
    """Exécute l'application Streamlit."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🏀")
    st.title(APP_TITLE)
    st.caption(f"Assistant virtuel pour {NAME} | Modèle : {MODEL_NAME}")
    st.caption(
        "Questions chiffrées → base SQL (players/stats). "
        "Questions d'analyse → rapports PDF (RAG)."
    )

    vector_store_manager = get_vector_store_manager()
    if vector_store_manager is None:
        st.error(
            "L'index documentaire est indisponible. Exécutez `python indexer.py` "
            "depuis la racine du projet avant de lancer l'application."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"Bonjour ! Je suis votre analyste IA pour la {NAME}. "
                    "Posez-moi une question chiffrée (statistiques) ou une question "
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

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            try:
                _, response_content = afficher_reponse(prompt, vector_store_manager)
            except Exception as error:
                LOGGER.exception("Erreur lors du traitement de la question")
                if logfire:
                    logfire.error("Erreur pipeline agent", error=str(error))
                response_content = (
                    "Une erreur technique empêche la génération de la réponse : "
                    f"{error}"
                )
                st.error(response_content)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_content}
    )


if __name__ == "__main__":
    main()
