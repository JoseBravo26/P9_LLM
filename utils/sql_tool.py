"""Tool LangChain SQL : le LLM propose une requete, cette couche la controle avant execution.

Regles de securite appliquees dans validate_sql :
    - seule une requete SELECT unique est acceptee ;
    - les mots-cles de modification/administration sont interdits ;
    - un LIMIT par defaut est ajoute si absent, pour bornancer la volumetrie renvoyee.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

from few_shot_sql_examples import FEW_SHOT_SQL
from utils.config import MODEL_NAME, require_mistral_api_key
from utils.database import DEFAULT_DB_PATH, get_connection
from utils.db_schemas import SQLRequest, SQLToolResult

LOGGER = logging.getLogger(__name__)
SCHEMA_TEXT = Path("db/schema.sql").read_text(encoding="utf-8")
FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|GRANT|REVOKE)\b|;",
    re.IGNORECASE,
)
DEFAULT_ROW_LIMIT = 20


def validate_sql(sql_text: str) -> str:
    """Refuse toute requete non-SELECT ou multi-instructions, et plafonne le volume."""
    candidate = sql_text.strip().rstrip(";")
    if not candidate.upper().startswith("SELECT"):
        raise ValueError("Seules les requetes SELECT sont autorisees.")
    if FORBIDDEN_PATTERN.search(candidate):
        raise ValueError("Requete refusee : mot-cle interdit ou instructions multiples detectees.")
    if "LIMIT" not in candidate.upper():
        candidate = f"{candidate} LIMIT {DEFAULT_ROW_LIMIT}"
    return candidate


def build_prompt(question: str) -> str:
    """Construit le prompt few-shot envoye au LLM pour generer le SQL."""
    examples_text = "\n".join(
        f"Question: {example['question']}\nSQL: {example['sql']}" for example in FEW_SHOT_SQL
    )
    return (
        "Tu generes uniquement une requete SQLite SELECT en une seule instruction, "
        "sans balise markdown et sans point-virgule final.\n"
        "Si la question porte sur une granularite absente du schema "
        "(match par match, cinq derniers matchs, domicile/exterieur), "
        "reponds exactement : ABSTAIN\n\n"
        f"SCHEMA DE LA BASE:\n{SCHEMA_TEXT}\n\n"
        f"EXEMPLES:\n{examples_text}\n\n"
        f"QUESTION: {question}\nSQL:"
    )


def generate_sql(question: str) -> str:
    """Interroge le LLM Mistral pour obtenir le SQL candidat associe a la question."""
    llm = ChatMistralAI(model=MODEL_NAME, mistral_api_key=require_mistral_api_key(), temperature=0)
    response = llm.invoke(build_prompt(question))
    return response.content.strip()


def execute_sql(question: str, db_path=DEFAULT_DB_PATH) -> SQLToolResult:
    """Genere, valide puis execute le SQL. Retourne toujours un resultat structure."""
    generated_sql = generate_sql(question)
    if generated_sql.upper().startswith("ABSTAIN"):
        return SQLToolResult(
            sql="",
            columns=[],
            rows=[],
            row_count=0,
            error="Les donnees structurees disponibles ne permettent pas cette granularite (saison uniquement).",
        )
    try:
        safe_sql = validate_sql(generated_sql)
        request = SQLRequest(question=question, sql=safe_sql)
        with get_connection(db_path) as connection:
            cursor_rows = connection.execute(request.sql).fetchall()
        rows = [dict(row) for row in cursor_rows]
        columns = list(rows[0].keys()) if rows else []
        return SQLToolResult(sql=request.sql, columns=columns, rows=rows, row_count=len(rows))
    except Exception as exc:
        LOGGER.warning("Echec d execution SQL pour la question '%s' : %s", question, exc)
        return SQLToolResult(sql=generated_sql, columns=[], rows=[], row_count=0, error=str(exc))


@tool
def nba_sql_tool(question: str) -> dict:
    """Interroge les statistiques NBA structurees (players/stats) pour repondre aux questions chiffrees et comparatives."""
    return execute_sql(question).model_dump()
