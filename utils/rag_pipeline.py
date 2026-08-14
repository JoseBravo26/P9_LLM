# Chaîne unique appelée par Streamlit et par l'évaluateur : traçage + sortie structurée.
import logging
from typing import Optional

from pydantic_ai import Agent

from .config import MODEL_NAME
from .schemas import RAGQuery, AssistantAnswer, PipelineResult
from .vector_store import VectorStoreManager

# Configuration de Logfire
logfire: Optional[object] = None
try:
    import logfire as lf
    lf.configure()
    lf.info("Pipeline RAG initialisé avec Logfire")
    logfire = lf
except ImportError:
    logging.warning("Logfire non installé : désactivation du traçage")
except Exception as e:
    logging.warning("Logfire indisponible : %s", e)

PROMPT = """Tu es analyste NBA. Utilise exclusivement le CONTEXTE. 
Si une donnée est absente, notamment détail par match ou domicile/extérieur, 
indique clairement que tu ne peux pas répondre. Cite les chunk_ids utilisés."""

agent = Agent(f'mistral:{MODEL_NAME}', output_type=AssistantAnswer, system_prompt=PROMPT)


def answer_question(question: str, top_k: int = 5, store: Optional[VectorStoreManager] = None) -> PipelineResult:
    """
    Exécute le pipeline RAG complet avec traçage Logfire.
    
    Args:
        question: La question de l'utilisateur
        top_k: Nombre de chunks à récupérer
        store: Instance VectorStoreManager (optionnelle, crée une nouvelle si None)
    
    Returns:
        PipelineResult avec la query, les contexts et la réponse
    """
    query = RAGQuery(question=question, top_k=top_k)
    store = store or VectorStoreManager()
    
    # Récupération des chunks
    contexts = store.search(query.question, query.top_k)
    
    # Construction du contexte
    context = '\n\n'.join(
        f'[chunk_id={x.id}; source={x.metadata.source}]\n{x.text}' 
        for x in contexts
    )
    
    # Définition de la fonction d'appel à l'agent
    def call_agent():
        return agent.run_sync(f'CONTEXTE:\n{context}\n\nQUESTION: {query.question}')
    
    # Exécution avec ou sans traçage Logfire
    if logfire is None:
        result = call_agent()
    else:
        import logfire as lf
        with lf.span(
            'rag.answer',
            question=query.question,
            retrieved_chunks=len(contexts),
            top_k=top_k,
        ) as span:
            result = call_agent()
            span.set_attribute('response_tokens', result.usage.total_tokens if result.usage else 0)
            lf.info(
                "Réponse générée",
                tokens_total=result.usage.total_tokens if result.usage else 0,
                n_contexts=len(contexts),
            )
    
    return PipelineResult(query=query, contexts=contexts, response=result.output)