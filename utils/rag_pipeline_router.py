"""Routeur agent : dirige chaque question vers le SQL Tool (chiffre) ou le RAG (narratif).

Ce module orchestre les deux chaines existantes :
    - utils/sql_tool.py       pour les questions statistiques/comparatives ;
    - utils/rag_pipeline.py   pour les questions narratives sur les rapports PDF.
Il expose une fonction unique `answer(question)` que l'interface Streamlit peut
appeler sans connaitre le detail du routage.

Correctif du 02/09/2026 (evaluation RAGAS incomplete) :
    Le script evaluate_ragas.py appelait directement
    utils.rag_pipeline.answer_question(), en contournant entierement ce
    routeur. Consequence : TOUTES les questions du dataset d'evaluation,
    y compris celles necessitant clairement la route SQL (ex: pourcentage a
    3 points de Nikola Jokic), etaient forcees vers le RAG documentaire seul,
    ce qui ne reflete pas le comportement reel de l'application (MistralChat.py
    utilise bien ce routeur). evaluate_ragas.py a ete corrige pour appeler
    answer() ci-dessous a la place.

    Pour que cette correction soit utile aux metriques RAGAS (qui ont besoin
    du contexte documentaire complet, pas seulement d'identifiants de chunks),
    answer_with_rag() et answer() exposent desormais aussi la liste des
    chunks documentaires complets (PipelineResult.contexts) sous la cle
    "contexts" du dictionnaire retourne par answer(). Cette cle est absente
    (liste vide) pour la branche SQL, qui n'a pas de chunks documentaires --
    son "contexte" reel est le resultat SQL lui-meme, deja trace via Logfire
    et disponible dans les logs (attribut 'sql' du span 'router.sql_branch').
    L'ajout de cette cle est retro-compatible : MistralChat.py, qui ne lit
    que answer()["route"] et answer()["response"], continue de fonctionner
    sans aucune modification necessaire de son cote.
"""
from __future__ import annotations

import logging
from typing import Optional

from pydantic_ai import Agent

from utils.config import MODEL_NAME
from utils.schemas import AssistantAnswer, PipelineResult, RAGQuery, RetrievedChunk
from utils.sql_tool import execute_sql
from utils.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

try:
    import logfire
except ImportError:
    logfire = None

ROUTER_PROMPT = """Tu classes une question sur une équipe NBA en une seule étiquette parmi :
- SQL : question chiffrée, comparative ou d’agrégation sur des statistiques de joueurs/équipes
  (exemples : pourcentage à 3 points, nombre de rebonds, comparaison entre joueurs, classement).
- RAG : question narrative, qualitative ou d’analyse issue de commentaires/rapports de match
  (exemples : analyse tactique, avis, contexte d’un match, débat entre fans).
- MIXED : question qui combine à la fois une demande chiffrée (statistiques, comparaison,
  classement…) ET une demande narrative (analyse, contexte, avis) dans la même phrase.

Règles :
- Réponds uniquement par SQL, RAG ou MIXED, sans aucun autre mot.
- Les formulations informelles et fautives restent des questions SQL dès qu’elles demandent
  une statistique, une comparaison ou un classement.
- Si la question contient clairement une partie “analyse” ET une partie “statistique”
  (par exemple « analyse la défense de Denver ET donne le pourcentage à 3 points de Jokic »),
  choisis systématiquement MIXED.

Exemples SQL :
- c lekel le + fort au shoot entre booker et tatum ?
- ki a le + de rebonds entre jokic giannis et KAT ?

Exemples RAG :
- Que disent les rapports sur la défense de Denver ?
- Comment les fans analysent la saison de Julius Randle ?

Exemples MIXED :
- Analyse la défense de Denver et donne le pourcentage à 3 points de Nikola Jokic.
- Explique la saison de Minnesota et compare les rebonds de Jokic et Towns.
"""

SQL_SYNTHESIS_PROMPT = """Tu es analyste NBA. Rédige une réponse concise et sourcée à partir du RESULTAT SQL fourni.

RÈGLES :
- Utilise exclusivement les lignes et colonnes présentes dans le RESULTAT SQL.
- Cite les colonnes et les valeurs utilisées.
- Si le résultat est vide ou contient une erreur, explique clairement que la donnée demandée n'est pas disponible dans les statistiques de saison.
- Ne calcule et n'invente jamais une statistique absente du résultat fourni.
- Pour toute comparaison de plusieurs joueurs, associe strictement chaque valeur au `player_name` figurant sur la même ligne du RESULTAT SQL.
- Ne déduis jamais l'association entre un joueur et une valeur à partir de l'ordre de la question.
- Avant de répondre, vérifie que le joueur présenté comme premier possède bien la valeur la plus élevée dans les lignes SQL, lorsque la question demande le meilleur, le plus élevé ou un classement.
- Les termes `stp`, `svp` et `merci` sont des marques de politesse ; ne les interprète jamais comme des statistiques.
- Chaque ligne est au format `colonne=valeur`.
- Pour une comparaison, recopie l'association `player_name=...` et
  `field_goal_pct=...` de chaque ligne exactement telle qu'elle est fournie.
- Si les lignes sont triées par une colonne décroissante, le premier joueur
  de la liste possède la valeur la plus élevée.
- N'inverse jamais des valeurs entre deux joueurs, même si leur ordre dans
  la QUESTION est différent.
  - Traduis les noms techniques de colonnes en français naturel.
- field_goal_pct doit être formulé « pourcentage de tirs réussis ».
- Affiche les pourcentages avec le symbole % et une virgule décimale.
- Ne mentionne pas les noms internes de colonnes SQL, sauf si l'utilisateur
  les demande explicitement.
"""

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
    """Classe la question en 'SQL', 'RAG' ou 'MIXED'. Repli sur RAG si la classification échoue."""
    with _traced_span("router.classify", question=question[:200]):
        try:
            decision = router_agent.run_sync(question).output.strip().upper()
        except Exception as exc:
            LOGGER.warning("Echec de classification, repli sur RAG : %s", exc)
            if logfire:
                logfire.error("Erreur routeur", error=str(exc))
            return "RAG"

    if "MIXED" in decision:
        route = "MIXED"
    elif "SQL" in decision:
        route = "SQL"
    else:
        route = "RAG"

    if logfire:
        logfire.info("Question routee", route=route, question=question[:200])
    return route


def answer_with_sql(question: str) -> AssistantAnswer:
    """Exécute le SQL Tool puis synthétise le résultat de façon contrôlée."""
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
                    "Je ne peux pas répondre avec les données structurées "
                    f"disponibles : {result.error}"
                ),
                cited_chunk_ids=[],
                confidence="high",
                abstained=True,
            )

        if result.row_count == 0:
            return AssistantAnswer(
                answer=(
                    "Je ne peux pas répondre avec les données structurées "
                    "disponibles : aucune donnée correspondante n’a été trouvée."
                ),
                cited_chunk_ids=[],
                confidence="high",
                abstained=True,
            )
        rows_text = "\n".join(
            " | ".join(f"{column}={value}" for column, value in row.items())
            for row in result.rows
        )

        prompt = (
            f"QUESTION: {question}\n\n"
            f"RESULTAT SQL — {result.row_count} ligne(s) :\n"
            f"{rows_text}\n\n"
            "RÈGLE ABSOLUE : associe une valeur uniquement au player_name de la "
            "même ligne. Ne change jamais l'association player_name/valeur."
        )

        return sql_synthesis_agent.run_sync(prompt).output


def answer_with_rag(
    question: str, top_k: int = 5, store: Optional[VectorStoreManager] = None
) -> tuple[AssistantAnswer, list[RetrievedChunk]]:
    """Delegue au pipeline RAG existant et retourne la reponse structuree
    ainsi que les chunks documentaires complets ayant servi de contexte.

    Le retour inclut desormais les chunks (et non plus seulement la reponse)
    afin que l'evaluation RAGAS puisse mesurer correctement la fidelite et
    la precision du contexte pour les questions routees vers le RAG.
    """
    from utils.rag_pipeline import answer_question
    with _traced_span("router.rag_branch", question=question[:200]):
        result: PipelineResult = answer_question(question, top_k=top_k, store=store)
        return result.response, result.contexts

def answer_mixed(
    question: str,
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,
) -> dict:
    """
    Traite une question MIXED en combinant la branche SQL et la branche RAG.

    - Partie SQL : utilise answer_with_sql(question) pour produire la réponse chiffrée.
    - Partie RAG : utilise answer_with_rag(question) pour produire la réponse narrative.
    - Construit une AssistantAnswer unique qui contient les deux parties clairement séparées.
    - Retourne également les réponses détaillées sql_answer et rag_answer pour l’analyse.
    """
    with _traced_span("router.mixed_branch", question=question[:200]):
        # 1) Réponse chiffrée
        sql_answer = answer_with_sql(question)

        # 2) Réponse narrative + contextes
        rag_answer, rag_contexts = answer_with_rag(question, top_k=top_k, store=store)

        # 3) Construction d’une réponse combinée lisible
        if sql_answer.abstained and rag_answer.abstained:
            combined_text = (
                "Je ne peux pas répondre de façon fiable ni sur la partie "
                "statistique, ni sur la partie analyse documentaire avec les "
                "sources disponibles."
            )
            combined_abstained = True
            combined_confidence = "high"
        else:
            # On explique clairement les deux parties
            combined_text = (
                "Partie statistiques (SQL) :\n"
                f"{sql_answer.answer}\n\n"
                "Partie analyse documentaire (RAG) :\n"
                f"{rag_answer.answer}"
            )
            # Si une des deux parties est fragile, on reste prudent
            combined_abstained = False
            combined_confidence = "medium"

        combined_answer = AssistantAnswer(
            answer=combined_text,
            cited_chunk_ids=rag_answer.cited_chunk_ids,
            confidence=combined_confidence,
            abstained=combined_abstained,
        )

        return {
            "route": "MIXED",
            "response": combined_answer,
            "contexts": rag_contexts,
            "sql_answer": sql_answer,
            "rag_answer": rag_answer,
        }

def answer(
    question: str,
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,
) -> dict:
    """
    Point d’entrée unique pour l’interface : route puis répond, avec métadonnées de tracage.

    Retourne un dictionnaire prêt à afficher :

    - "route": "SQL" | "RAG" | "MIXED",
    - "response": AssistantAnswer,
    - "contexts": list[RetrievedChunk]  # vide pour la branche SQL,
    - éventuellement "sql_answer" et "rag_answer" pour MIXED.
    """
    route = route_question(question)

    if route == "SQL":
        with _traced_span("router.sql_branch", question=question[:200]):
            response = answer_with_sql(question)
        return {"route": "SQL", "response": response, "contexts": []}

    if route == "RAG":
        response, contexts = answer_with_rag(question, top_k=top_k, store=store)
        return {"route": "RAG", "response": response, "contexts": contexts}

    # Route MIXED : combinaison des deux
    return answer_mixed(question, top_k=top_k, store=store)