"""Tests du routeur SQL/RAG/MIXED.

Ces tests vérifient :
- le contrat de sortie de la branche SQL ;
- le repli sur RAG si la classification échoue ;
- la détection correcte des routes SQL, RAG et MIXED ;
- l’assemblage d’une réponse mixte combinant SQL et RAG.
"""

from unittest.mock import MagicMock, patch

from utils.db_schemas import SQLToolResult
from utils.schemas import AssistantAnswer, RetrievedChunk, SourceMetadata


def test_answer_with_sql_retourne_abstention_si_erreur():
    """La branche SQL doit s'abstenir proprement si le SQL Tool remonte une erreur."""
    from utils.rag_pipeline_router import answer_with_sql

    with patch("utils.rag_pipeline_router.execute_sql") as mock_execute:
        mock_execute.return_value = SQLToolResult(
            sql="",
            columns=[],
            rows=[],
            row_count=0,
            error="Granularité indisponible.",
        )

        result = answer_with_sql("Quel joueur a le meilleur 3P% sur les 5 derniers matchs ?")

    assert isinstance(result, AssistantAnswer)
    assert result.abstained is True
    assert "Granularité indisponible" in result.answer


def test_route_question_replie_sur_rag_si_erreur_llm():
    """En cas d'échec du classifieur, le routeur doit replier sur RAG."""
    from utils.rag_pipeline_router import route_question

    with patch("utils.rag_pipeline_router.router_agent") as mock_agent:
        mock_agent.run_sync.side_effect = RuntimeError("Erreur réseau simulée")
        route = route_question("Question quelconque")

    assert route == "RAG"


def test_route_question_detecte_sql():
    """Une question statistique doit être classée en SQL."""
    from utils.rag_pipeline_router import route_question

    fake_result = MagicMock()
    fake_result.output = "SQL"

    with patch("utils.rag_pipeline_router.router_agent") as mock_agent:
        mock_agent.run_sync.return_value = fake_result
        route = route_question("Quel est le pourcentage à 3 points de Jokic ?")

    assert route == "SQL"


def test_route_question_detecte_rag():
    """Une question narrative doit être classée en RAG."""
    from utils.rag_pipeline_router import route_question

    fake_result = MagicMock()
    fake_result.output = "RAG"

    with patch("utils.rag_pipeline_router.router_agent") as mock_agent:
        mock_agent.run_sync.return_value = fake_result
        route = route_question("Que disent les rapports sur la défense de Denver ?")

    assert route == "RAG"


def test_route_question_detecte_mixed():
    """Une question combinant analyse et statistique doit être classée en MIXED."""
    from utils.rag_pipeline_router import route_question

    fake_result = MagicMock()
    fake_result.output = "MIXED"

    with patch("utils.rag_pipeline_router.router_agent") as mock_agent:
        mock_agent.run_sync.return_value = fake_result
        route = route_question(
            "Analyse la défense de Denver et donne le pourcentage à 3 points de Nikola Jokic."
        )

    assert route == "MIXED"


def test_answer_mixed_combine_les_reponses_sql_et_rag():
    """La branche MIXED doit assembler les réponses SQL et RAG dans une réponse unique."""
    from utils.rag_pipeline_router import answer_mixed

    sql_answer = AssistantAnswer(
        answer="Nikola Jokic a un pourcentage à 3 points de 35,9 %.",
        cited_chunk_ids=[],
        confidence="high",
        abstained=False,
    )

    rag_answer = AssistantAnswer(
        answer="Les rapports décrivent une défense de Denver structurée et disciplinée.",
        cited_chunk_ids=["chunk_1"],
        confidence="medium",
        abstained=False,
    )

    fake_contexts = [
        RetrievedChunk(
            id="chunk_1",
            text="Denver défend de manière disciplinée avec une bonne protection du cercle.",
            score=0.91,
            metadata=SourceMetadata(
                source="inputs/Reddit 1.pdf",
                filename="Reddit 1.pdf",
                category="match_report",
            ),
        )
    ]

    with patch("utils.rag_pipeline_router.answer_with_sql", return_value=sql_answer):
        with patch("utils.rag_pipeline_router.answer_with_rag", return_value=(rag_answer, fake_contexts)):
            result = answer_mixed(
                "Analyse la défense de Denver et donne le pourcentage à 3 points de Nikola Jokic."
            )

    assert result["route"] == "MIXED"
    assert isinstance(result["response"], AssistantAnswer)
    assert result["response"].abstained is False
    assert "Partie statistiques" in result["response"].answer
    assert "Nikola Jokic a un pourcentage à 3 points de 35,9 %." in result["response"].answer
    assert "Partie analyse documentaire" in result["response"].answer
    assert "Les rapports décrivent une défense de Denver structurée et disciplinée." in result["response"].answer
    assert result["contexts"] == fake_contexts
    assert result["sql_answer"] == sql_answer
    assert result["rag_answer"] == rag_answer
    assert result["response"].cited_chunk_ids == ["chunk_1"]


def test_answer_mixed_abstention_totale_si_sql_et_rag_abstiennent():
    """Si les deux branches s'abstiennent, la réponse MIXED doit s'abstenir aussi."""
    from utils.rag_pipeline_router import answer_mixed

    sql_answer = AssistantAnswer(
        answer="Je ne peux pas répondre avec les données structurées disponibles.",
        cited_chunk_ids=[],
        confidence="high",
        abstained=True,
    )

    rag_answer = AssistantAnswer(
        answer="Je ne peux pas répondre avec les sources disponibles.",
        cited_chunk_ids=[],
        confidence="high",
        abstained=True,
    )

    with patch("utils.rag_pipeline_router.answer_with_sql", return_value=sql_answer):
        with patch("utils.rag_pipeline_router.answer_with_rag", return_value=(rag_answer, [])):
            result = answer_mixed("Question mixte impossible à traiter")

    assert result["route"] == "MIXED"
    assert result["response"].abstained is True
    assert "Je ne peux pas répondre de façon fiable" in result["response"].answer


def test_answer_route_vers_mixed_appelle_answer_mixed():
    """Le point d'entrée principal doit déléguer à answer_mixed si la route vaut MIXED."""
    from utils.rag_pipeline_router import answer

    combined_answer = AssistantAnswer(
        answer="Réponse combinée.",
        cited_chunk_ids=["chunk_1"],
        confidence="medium",
        abstained=False,
    )

    mixed_result = {
        "route": "MIXED",
        "response": combined_answer,
        "contexts": [],
        "sql_answer": AssistantAnswer(
            answer="Réponse SQL.",
            cited_chunk_ids=[],
            confidence="high",
            abstained=False,
        ),
        "rag_answer": AssistantAnswer(
            answer="Réponse RAG.",
            cited_chunk_ids=["chunk_1"],
            confidence="medium",
            abstained=False,
        ),
    }

    with patch("utils.rag_pipeline_router.route_question", return_value="MIXED"):
        with patch("utils.rag_pipeline_router.answer_mixed", return_value=mixed_result) as mock_mixed:
            result = answer("Question mixte")

    mock_mixed.assert_called_once_with("Question mixte", top_k=5, store=None)
    assert result["route"] == "MIXED"
    assert result["response"].answer == "Réponse combinée."