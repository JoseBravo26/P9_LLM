"""Tests du routeur SQL/RAG : verifient le contrat de sortie et le repli en cas d erreur."""
from unittest.mock import MagicMock, patch

from utils.schemas import AssistantAnswer
from utils.db_schemas import SQLToolResult


def test_answer_with_sql_retourne_abstention_si_erreur():
    from utils.rag_pipeline_router import answer_with_sql

    with patch("utils.rag_pipeline_router.execute_sql") as mock_execute:
        mock_execute.return_value = SQLToolResult(
            sql="", columns=[], rows=[], row_count=0, error="Granularite indisponible."
        )
        result = answer_with_sql("Quel joueur a le meilleur 3P% sur les 5 derniers matchs ?")

    assert isinstance(result, AssistantAnswer)
    assert result.abstained is True
    assert "Granularite indisponible" in result.answer


def test_route_question_replie_sur_rag_si_erreur_llm():
    from utils.rag_pipeline_router import route_question

    with patch("utils.rag_pipeline_router.router_agent") as mock_agent:
        mock_agent.run_sync.side_effect = RuntimeError("Erreur reseau simulee")
        route = route_question("Question quelconque")

    assert route == "RAG"


def test_route_question_detecte_sql():
    from utils.rag_pipeline_router import route_question

    fake_result = MagicMock()
    fake_result.output = "SQL"
    with patch("utils.rag_pipeline_router.router_agent") as mock_agent:
        mock_agent.run_sync.return_value = fake_result
        route = route_question("Quel est le pourcentage a 3 points de Jokic ?")

    assert route == "SQL"
