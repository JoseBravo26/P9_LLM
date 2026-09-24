"""Pipeline RAG narratif pour NBA Analyst AI.

Ce module est utilisé :
- par l'application Streamlit ;
- par le routeur hybride SQL/RAG/MIXED ;
- par le script d'évaluation RAGAS.

Objectifs de cette version corrigée :
- utiliser exclusivement le contexte documentaire récupéré ;
- détecter plus proprement les contextes hors sujet ;
- produire une abstention explicite quand les sources sont insuffisantes ;
- conserver une sortie structurée compatible avec le reste du projet.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic_ai import Agent

from .config import MODEL_NAME
from .schemas import AssistantAnswer, PipelineResult, RAGQuery
from .vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

# Configuration optionnelle de Logfire
logfire: Optional[object] = None
try:
    import logfire as lf

    lf.configure()
    lf.info("Pipeline RAG initialisé avec Logfire")
    logfire = lf
except ImportError:
    LOGGER.warning("Logfire non installé : désactivation du traçage")
except Exception as error:
    LOGGER.warning("Logfire indisponible : %s", error)


PROMPT = """
Tu es un analyste NBA spécialisé dans la synthèse de rapports textuels.

Tu dois répondre exclusivement à partir des passages présents dans le CONTEXTE.

Règles obligatoires :

1. Vérifie d'abord que le CONTEXTE parle réellement du sujet demandé.
2. Ne considère pas un passage comme pertinent simplement parce qu'il contient
   des mots généraux comme « défense », « équipe », « match », « playoffs »
   ou « NBA ».
3. Pour une question sur une équipe, vérifie que cette équipe est explicitement
   mentionnée dans les passages que tu utilises.
4. Pour une question sur un joueur, vérifie que ce joueur est explicitement
   mentionné dans les passages que tu utilises.
5. Si les passages parlent surtout d'autres équipes, d'autres joueurs,
   des médias, du public, d'un débat général ou d'un sujet voisin,
   considère le contexte comme insuffisant.
6. Si le contexte est insuffisant, ambigu ou hors sujet, réponds exactement :
   « Je ne peux pas répondre avec les sources disponibles. »
7. N'utilise jamais tes connaissances générales sur la NBA pour compléter
   une information absente du CONTEXTE.
8. Ne déduis jamais une analyse spécifique à partir d'un passage général.
9. Cite uniquement les chunk_ids réellement utilisés pour soutenir la réponse.
10. Si tu t'abstiens, cited_chunk_ids doit être une liste vide.
11. confidence doit valoir :
    - high si les passages soutiennent clairement la réponse ;
    - medium si les passages sont partiels mais suffisants ;
    - low si les passages sont fragiles ;
12. abstained doit valoir true uniquement si tu ne peux pas répondre
    avec les sources disponibles.

Format attendu :
- answer : texte final en français ;
- cited_chunk_ids : liste des chunks réellement utilisés ;
- confidence : high, medium ou low ;
- abstained : true ou false.
""".strip()

agent = Agent(
    f"mistral:{MODEL_NAME}",
    output_type=AssistantAnswer,
    system_prompt=PROMPT,
)


def _build_context_text(contexts: list) -> str:
    """Assemble les chunks récupérés dans un contexte textuel lisible par le LLM."""
    return "\n\n".join(
        f"[chunk_id={chunk.id}; source={chunk.metadata.source}]\n{chunk.text}"
        for chunk in contexts
    )


def _build_abstention_answer() -> AssistantAnswer:
    """Construit une abstention standard lorsque le contexte est vide ou inutilisable."""
    return AssistantAnswer(
        answer="Je ne peux pas répondre avec les sources disponibles.",
        cited_chunk_ids=[],
        confidence="high",
        abstained=True,
    )


def _call_rag_agent(question: str, context: str) -> AssistantAnswer:
    """Appelle l'agent RAG avec un prompt d'exécution strict."""
    execution_prompt = (
        "CONTEXTE DOCUMENTAIRE :\n"
        f"{context}\n\n"
        "QUESTION UTILISATEUR :\n"
        f"{question}\n\n"
        "Consigne finale : avant de répondre, vérifie que les passages du contexte "
        "traitent directement le sujet demandé. Si ce n'est pas le cas, abstiens-toi "
        "exactement avec la formule imposée."
    )

    return agent.run_sync(execution_prompt).output


def answer_question(
    question: str,
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,
) -> PipelineResult:
    """
    Exécute le pipeline RAG complet avec sortie structurée.

    Étapes :
    1. Validation et normalisation de la question ;
    2. Recherche vectorielle des chunks les plus pertinents ;
    3. Construction d'un contexte documentaire ;
    4. Génération d'une réponse structurée ou d'une abstention explicite.

    Args:
        question: Question utilisateur.
        top_k: Nombre maximal de chunks à récupérer.
        store: Gestionnaire de vector store optionnel.

    Returns:
        PipelineResult contenant la requête, les chunks récupérés et la réponse.
    """
    query = RAGQuery(question=question, top_k=top_k)
    store = store or VectorStoreManager()

    contexts = store.search(query.question, query.top_k)

    if not contexts:
        LOGGER.info(
            "Aucun chunk pertinent trouvé pour la question : %s",
            query.question,
        )

        if logfire:
            import logfire as lf

            lf.info(
                "Aucun contexte RAG retenu",
                question=query.question,
                top_k=top_k,
            )

        return PipelineResult(
            query=query,
            contexts=[],
            response=_build_abstention_answer(),
        )

    context_text = _build_context_text(contexts)

    def call_agent() -> AssistantAnswer:
        return _call_rag_agent(query.question, context_text)

    if logfire is None:
        response = call_agent()
    else:
        import logfire as lf

        with lf.span(
            "rag.answer",
            question=query.question,
            retrieved_chunks=len(contexts),
            top_k=top_k,
        ) as span:
            response = call_agent()
            span.set_attribute(
                "response_tokens",
                0,
            )
            lf.info(
                "Réponse RAG générée",
                abstained=response.abstained,
                confidence=response.confidence,
                cited_chunks=len(response.cited_chunk_ids),
                retrieved_chunks=len(contexts),
            )

    return PipelineResult(
        query=query,
        contexts=contexts,
        response=response,
    )