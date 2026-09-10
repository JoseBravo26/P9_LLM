"""Pipeline d ingestion : Excel de statistiques + PDF de commentaires -> SQLite.

Exemple d execution :
    python load_excel_to_db.py --excel "inputs/regular NBA.xlsx" --reports-dir inputs --season 2024-2025

Correctif du 18/08/2026 (en-tete 3PM) :
    La colonne d en-tete "3PM" (tirs a 3 points reussis) du classeur Excel
    etait automatiquement convertie par openpyxl en objet `datetime.time`
    (3PM etant interprete comme "3 heures de l'apres-midi", soit 15h00),
    plutot que d etre conservee comme la chaine de caracteres "3PM". Le
    dictionnaire COLUMN_MAP cherche une correspondance exacte sur la chaine
    "3PM" : comme l en-tete reel n etait pas une chaine mais un objet time,
    aucun renommage ne se produisait, et la colonne `three_point_made`
    n etait jamais peuplee (toujours NULL) pour l ensemble des joueurs. La
    ligne `dataframe.columns = [str(col) for col in dataframe.columns]`
    force chaque en-tete de colonne en chaine de caracteres immediatement
    apres la lecture du fichier, avant tout renommage. La colonne devient
    alors la chaine "15:00:00", d ou l ajout de cette cle dans COLUMN_MAP.

Correctif du 18/08/2026 (upsert et index unique partiel) :
    Un index unique partiel a ete ajoute dans db/schema.sql pour empecher
    les doublons de statistiques de saison (le match_id etant toujours NULL
    dans ce cas, l'ancienne contrainte UNIQUE a 4 colonnes ne detectait
    jamais de conflit, car NULL n'est jamais egal a NULL en SQL) :
        CREATE UNIQUE INDEX idx_stats_season_unique
        ON stats(player_id, season_label, granularity)
        WHERE match_id IS NULL;
    La clause ON CONFLICT de upsert_player_and_stats() cible desormais
    exactement les colonnes de cet index (player_id, season_label,
    granularity) WHERE match_id IS NULL, comme l'exige SQLite.

Correctif du 18/08/2026 (extraction PDF sans OCR/torch) :
    Les 4 rapports PDF (exports de discussions Reddit) contiennent du texte
    natif extractible (verifie : plusieurs dizaines de milliers de
    caracteres par fichier), mais PyPDF2.extract_text() renvoyait moins de
    100 caracteres exploitables sur ces mises en page complexes (colonnes,
    encarts sponsorises, melange FR/EN), ce qui declenchait a tort le repli
    OCR. Ce repli echouait ensuite car easyocr necessite torch, absent du
    venv. Aucun GPU ni installation de torch n'est necessaire : le texte est
    deja present nativement dans les PDF. La fonction extract_pdf_text()
    utilise desormais PyMuPDF (module `fitz`, deja une dependance du projet
    via utils/data_loader.py) en premier recours, PyPDF2 en repli si
    PyMuPDF est indisponible, et l'OCR uniquement en tout dernier recours
    si aucune extraction native n'a fonctionne.
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
# Note : la cle "15:00:00" couvre le cas ou l'en-tete "3PM" a ete corrompu par
# openpyxl en objet datetime.time(15, 0), puis converti en chaine "15:00:00"
# par le correctif dataframe.columns = [str(c) for c in dataframe.columns].
COLUMN_MAP = {
    "Player": "player_name", "Team": "team_code", "Age": "age",
    "GP": "games_played", "W": "wins", "L": "losses", "Min": "minutes_per_game",
    "PTS": "total_points", "FG%": "field_goal_pct",
    "3PM": "three_point_made", "15:00:00": "three_point_made",
    "3PA": "three_point_attempted", "3P%": "three_point_pct",
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

    # Correctif : force chaque en-tete de colonne en chaine de caracteres.
    # Sans cette conversion, openpyxl peut interpreter un en-tete comme
    # "3PM" en objet datetime.time (15h00), ce qui empeche tout renommage
    # via COLUMN_MAP (qui compare des chaines exactes) et laisse la colonne
    # metier correspondante (three_point_made) totalement vide en base.
    dataframe.columns = [str(colonne) for colonne in dataframe.columns]

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
    """Insere ou met a jour un joueur, puis sa ligne de statistiques saisonnieres.

    Remarque sur l idempotence :
        Pour les statistiques de saison (granularity='season'), match_id est
        toujours NULL. La clause ON CONFLICT ci-dessous cible donc l'index
        unique partiel `idx_stats_season_unique` defini dans db/schema.sql
        sur (player_id, season_label, granularity) WHERE match_id IS NULL.
        SQLite exige que la clause ON CONFLICT(...) reference exactement les
        colonnes d'un index ou d'une contrainte unique existante : c'est
        pourquoi la clause ci-dessous porte sur (player_id, season_label,
        granularity) et non sur (player_id, match_id, season_label,
        granularity) comme dans l'ancienne version de ce fichier (qui ne
        correspondait plus a aucun index apres l'ajout de idx_stats_season_unique).
    """
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
        ON CONFLICT(player_id, season_label, granularity) WHERE match_id IS NULL
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


def _extraire_texte_pymupdf(pdf_path: Path) -> tuple[str, int]:
    """Extrait le texte natif d'un PDF via PyMuPDF (fitz), plus robuste que
    PyPDF2 sur les mises en page complexes (colonnes, encarts publicitaires,
    contenu multilingue) telles que les exports de pages web Reddit.
    Retourne une chaine vide si PyMuPDF n'est pas installe, pour permettre
    un repli transparent sur PyPDF2.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        return "", 0

    document = fitz.open(str(pdf_path))
    try:
        texte = "\n".join(page.get_text() for page in document)
        return texte.strip(), document.page_count
    finally:
        document.close()


def _extraire_texte_pypdf2(pdf_path: Path) -> tuple[str, int]:
    """Extrait le texte natif d'un PDF via PyPDF2 (repli historique)."""
    reader = PdfReader(str(pdf_path))
    texte = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return texte, len(reader.pages)


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """Extrait le texte d'un PDF en essayant plusieurs strategies dans l'ordre :

        1. PyMuPDF (fitz) : le plus robuste sur les mises en page complexes,
           evite de recourir a l'OCR pour des PDF contenant deja du texte
           natif mais mal gere par PyPDF2 (ex: exports de pages web Reddit).
        2. PyPDF2 : repli si PyMuPDF n'est pas installe.
        3. OCR (EasyOCR) : dernier recours si aucune extraction native n'a
           produit suffisamment de texte (ex: PDF scanne, image pure) ;
           necessite les dependances optionnelles torch/PyMuPDF/Pillow.
    """
    texte, nb_pages = _extraire_texte_pymupdf(pdf_path)
    if len(texte) >= 100:
        return texte, nb_pages

    texte_pypdf2, nb_pages_pypdf2 = _extraire_texte_pypdf2(pdf_path)
    if len(texte_pypdf2) >= 100:
        return texte_pypdf2, nb_pages_pypdf2

    from utils.data_loader import extract_text_from_pdf_with_ocr

    nb_pages_finales = nb_pages or nb_pages_pypdf2
    ocr_text = extract_text_from_pdf_with_ocr(str(pdf_path)) or ""
    return ocr_text.strip(), nb_pages_finales


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
