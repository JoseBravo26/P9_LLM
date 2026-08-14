"""Routeur agent : dirige chaque question vers le SQL Tool (chiffre) ou le RAG (narratif).

Ce module orchestre les deux chaines existantes :
    - utils/sql_tool.py       pour les questions statistiques/comparatives ;
    - utils/rag_pipeline.py   pour les questions narratives sur les rapports PDF.
Il expose une fonction unique `answer(question)` que l'interface Streamlit peut
appeler sans connaitre le detail du routage.
"""
from __future__ import annotations

import logging
from typing import Optional

from pydantic_ai import Agent

from utils.config import MODEL_NAME
from utils.schemas import AssistantAnswer, PipelineResult, RAGQuery
from utils.sql_tool import execute_sql
from utils.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

try:
    import logfire
except ImportError:
    logfire = None

ROUTER_PROMPT = """Tu classes une question sur une equipe NBA en une seule etiquette parmi :
- SQL : question chiffree, comparative ou d agregation sur des statistiques de joueurs/equipes
  (exemples : pourcentage a 3 points, nombre de rebonds, comparaison entre joueurs, classement).
- RAG : question narrative, qualitative ou d analyse issue de commentaires/rapports de match
  (exemples : analyse tactique, avis, contexte d un match, debat entre fans).
Reponds uniquement par SQL ou RAG, sans aucun autre mot."""

SQL_SYNTHESIS_PROMPT = """Tu es analyste NBA. Redige une reponse concise et sourcee a partir
du RESULTAT SQL fourni. Cite les colonnes et valeurs utilisees. Si le resultat est vide ou
contient une erreur, explique clairement que la donnee demandee n est pas disponible dans les
statistiques de saison. Ne calcule et n invente jamais une statistique absente du resultat fourni."""

router_agent = Agent(f"mistral:{MODEL_NAME}", output_type=str, system_prompt=ROUTER_PROMPT)
sql_synthesis_agent = Agent(
    f"mistral:{MODEL_NAME}", output_type=AssistantAnswer, system_prompt=SQL_SYNTHESIS_PROMPT
)


def _traced_span(name: str, **attributes):
    """Retourne un span Logfire si disponible, sinon un context manager neutre."""
    if logfire is not None:
        return logfire.span(name, **attributes)
    from contextlib import nullcontext
    return nullcontext()


def route_question(question: str) -> str:
    """Classe la question en 'SQL' ou 'RAG'. Repli sur RAG si la classification echoue."""
    with _traced_span("router.classify", question=question[:200]):
        try:
            decision = router_agent.run_sync(question).output.strip().upper()
        except Exception as exc:
            LOGGER.warning("Echec de classification, repli sur RAG : %s", exc)
            if logfire:
                logfire.error("Erreur routeur", error=str(exc))
            return "RAG"
    route = "SQL" if "SQL" in decision else "RAG"
    if logfire:
        logfire.info("Question routee", route=route, question=question[:200])
    return route


def answer_with_sql(question: str) -> AssistantAnswer:
    """Execute le SQL Tool puis fait synthetiser le resultat par le LLM."""
    with _traced_span("router.sql_branch", question=question[:200]):
        result = execute_sql(question)
        if logfire:
            logfire.info(
                "Resultat SQL Tool",
                row_count=result.row_count,
                has_error=bool(result.error),
                sql=result.sql[:300],
            )
        if result.error:
            return AssistantAnswer(
                answer=(
                    "Je ne peux pas repondre avec les donnees structurees disponibles : "
                    f"{result.error}"
                ),
                cited_chunk_ids=[],
                confidence="low",
                abstained=True,
            )
        prompt = f"QUESTION: {question}\nRESULTAT SQL ({result.row_count} lignes): {result.rows}"
        synthesis = sql_synthesis_agent.run_sync(prompt)
        return synthesis.output


def answer_with_rag(question: str, top_k: int = 5, store: Optional[VectorStoreManager] = None) -> AssistantAnswer:
    """Delegue au pipeline RAG existant et retourne uniquement la reponse structuree."""
    from utils.rag_pipeline import answer_question
    with _traced_span("router.rag_branch", question=question[:200]):
        result: PipelineResult = answer_question(question, top_k=top_k, store=store)
        return result.response


def answer(question: str, top_k: int = 5, store: Optional[VectorStoreManager] = None) -> dict:
    """Point d entree unique pour l interface : route puis repond, avec metadonnees de tracage.

    Retourne un dictionnaire pret a afficher :
        {"route": "SQL"|"RAG", "response": AssistantAnswer}
    """
    RAGQuery(question=question)  # Validation Pydantic de la question avant tout traitement.
    route = route_question(question)
    if route == "SQL":
        response = answer_with_sql(question)
    else:
        response = answer_with_rag(question, top_k=top_k, store=store)
    return {"route": route, "response": response}
