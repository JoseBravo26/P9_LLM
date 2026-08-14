"""Pipeline d ingestion : Excel de statistiques + PDF de commentaires -> SQLite.

Exemple d execution :
    python load_excel_to_db.py --excel "inputs/regular NBA.xlsx" --reports-dir inputs --season 2024-2025
"""
from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from pathlib import Path

import pandas as pd
from PyPDF2 import PdfReader
from pydantic import ValidationError

from utils.database import DEFAULT_DB_PATH, get_connection, initialize_database
from utils.db_schemas import SeasonStatRow

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

# Correspondance entre les en-tetes du classeur NBA et les noms de colonnes metier.
COLUMN_MAP = {
    "Player": "player_name", "Team": "team_code", "Age": "age",
    "GP": "games_played", "W": "wins", "L": "losses", "Min": "minutes_per_game",
    "PTS": "total_points", "FG%": "field_goal_pct",
    "3PM": "three_point_made", "3PA": "three_point_attempted", "3P%": "three_point_pct",
    "FT%": "free_throw_pct", "OREB": "offensive_rebounds", "DREB": "defensive_rebounds",
    "REB": "total_rebounds", "AST": "assists", "TOV": "turnovers", "STL": "steals",
    "BLK": "blocks", "OFFRTG": "offensive_rating", "DEFRTG": "defensive_rating",
    "NETRTG": "net_rating", "TS%": "true_shooting_pct", "USG%": "usage_pct",
}

REQUIRED_COLUMNS = {"player_name", "team_code", "games_played", "wins", "losses"}


def nullable(valeur):
    """Convertit les NaN pandas en None afin de rester compatible avec Pydantic."""
    if valeur is None:
        return None
    if isinstance(valeur, float) and math.isnan(valeur):
        return None
    if pd.isna(valeur):
        return None
    return valeur


def normalize_dataframe(excel_path: str, season_label: str) -> pd.DataFrame:
    """Charge le classeur Excel et renomme les colonnes vers le schema metier."""
    dataframe = pd.read_excel(excel_path, sheet_name="Données NBA", header=1)
    dataframe = dataframe.rename(columns=COLUMN_MAP)

    colonnes_manquantes = REQUIRED_COLUMNS - set(dataframe.columns)
    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes obligatoires manquantes dans l Excel : {sorted(colonnes_manquantes)}"
        )

    dataframe = dataframe.dropna(subset=["player_name"])
    dataframe["season_label"] = season_label
    return dataframe


def upsert_player_and_stats(connection: sqlite3.Connection, row: SeasonStatRow, source_file: str) -> None:
    """Insere ou met a jour un joueur, puis sa ligne de statistiques saisonnieres."""
    player_id = connection.execute(
        """
        INSERT INTO players (player_name, team_code, age)
        VALUES (?, ?, ?)
        ON CONFLICT(player_name) DO UPDATE SET team_code = excluded.team_code, age = excluded.age
        RETURNING player_id
        """,
        (row.player_name, row.team_code, row.age),
    ).fetchone()[0]

    stat_fields = [
        "games_played", "wins", "losses", "minutes_per_game", "total_points",
        "field_goal_pct", "three_point_made", "three_point_attempted", "three_point_pct",
        "free_throw_pct", "offensive_rebounds", "defensive_rebounds", "total_rebounds",
        "assists", "turnovers", "steals", "blocks", "offensive_rating",
        "defensive_rating", "net_rating", "true_shooting_pct", "usage_pct",
    ]
    values = [getattr(row, field) for field in stat_fields]
    all_columns = ["player_id", "match_id", "season_label", "granularity", *stat_fields, "source_file"]
    placeholders = ",".join("?" for _ in all_columns)
    update_clause = ",".join(f"{field} = excluded.{field}" for field in stat_fields)

    connection.execute(
        f"""
        INSERT INTO stats ({",".join(all_columns)})
        VALUES ({placeholders})
        ON CONFLICT(player_id, match_id, season_label, granularity)
        DO UPDATE SET {update_clause}, source_file = excluded.source_file
        """,
        [player_id, None, row.season_label, "season", *values, source_file],
    )


def ingest_excel(excel_path: str, season_label: str, db_path: str = DEFAULT_DB_PATH) -> tuple[int, int]:
    """Valide chaque ligne du classeur puis l alimente en base. Retourne (acceptees, rejetees)."""
    initialize_database(db_path)
    dataframe = normalize_dataframe(excel_path, season_label)
    accepted = rejected = 0
    with get_connection(db_path) as connection:
        for _, raw_row in dataframe.iterrows():
            payload = {key: nullable(value) for key, value in raw_row.to_dict().items()}
            try:
                validated_row = SeasonStatRow.model_validate(payload)
                upsert_player_and_stats(connection, validated_row, Path(excel_path).name)
                accepted += 1
            except (ValidationError, ValueError, sqlite3.Error) as exc:
                rejected += 1
                LOGGER.warning("Ligne rejetee (%s) : %s", payload.get("player_name", "?"), exc)
    LOGGER.info("Ingestion Excel terminee : %s lignes validees, %s rejetees.", accepted, rejected)
    return accepted, rejected


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """Extrait le texte d'un PDF, avec repli OCR si le texte natif est insuffisant."""
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if len(text) >= 100:
        return text, len(reader.pages)

    from utils.data_loader import extract_text_from_pdf_with_ocr

    ocr_text = extract_text_from_pdf_with_ocr(str(pdf_path)) or ""
    return ocr_text.strip(), len(reader.pages)


def ingest_reports(reports_dir: str, db_path: str = DEFAULT_DB_PATH) -> int:
    """Charge chaque PDF du dossier comme un rapport, sans invention de match associe."""
    initialize_database(db_path)
    inserted = 0
    with get_connection(db_path) as connection:
        for pdf_path in sorted(Path(reports_dir).glob("*.pdf")):
            try:
                text, page_count = extract_pdf_text(pdf_path)
                if not text:
                    LOGGER.warning("PDF sans texte exploitable, ignore : %s", pdf_path.name)
                    continue
                connection.execute(
                    """
                    INSERT INTO reports (title, content, source_file, page_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_file) DO UPDATE SET
                        title = excluded.title, content = excluded.content, page_count = excluded.page_count
                    """,
                    (pdf_path.stem, text, pdf_path.name, page_count),
                )
                inserted += 1
            except Exception as exc:
                LOGGER.warning("PDF rejete (%s) : %s", pdf_path.name, exc)
    LOGGER.info("Ingestion des rapports terminee : %s fichiers integres.", inserted)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion Excel + PDF vers SQLite (players, stats, reports).")
    parser.add_argument("--excel", required=True, help="Chemin du classeur de statistiques NBA.")
    parser.add_argument("--reports-dir", default="inputs", help="Dossier contenant les PDF de commentaires.")
    parser.add_argument("--season", default="2024-2025", help="Etiquette de saison a associer aux statistiques.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Chemin du fichier SQLite cible.")
    args = parser.parse_args()

    ingest_excel(args.excel, args.season, args.db)
    ingest_reports(args.reports_dir, args.db)


if __name__ == "__main__":
    main()
