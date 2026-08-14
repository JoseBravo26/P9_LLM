"""Connexion SQLite et initialisation idempotente du schema relationnel."""
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("database/nba_analytics.db")
SCHEMA_PATH = Path("db/schema.sql")


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Ouvre une connexion SQLite avec cles etrangeres actives et lignes nommees."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Cree les tables si elles n'existent pas encore. Operation sans risque a rejouer."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection(db_path) as connection:
        connection.executescript(schema_sql)
