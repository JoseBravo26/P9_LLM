"""Tests de la couche de securite SQL : seules les requetes SELECT bornees passent."""
import pytest

from utils.sql_tool import validate_sql


def test_select_simple_est_accepte_et_borne():
    sql = validate_sql("SELECT * FROM players")
    assert sql.upper().startswith("SELECT")
    assert "LIMIT" in sql.upper()


def test_limit_existant_est_preserve():
    sql = validate_sql("SELECT * FROM players LIMIT 5")
    assert sql.count("LIMIT") == 1


@pytest.mark.parametrize(
    "sql_dangereux",
    [
        "DELETE FROM players",
        "SELECT * FROM players; DROP TABLE players",
        "PRAGMA table_info(players)",
        "UPDATE players SET age = 0",
    ],
)
def test_requete_dangereuse_est_refusee(sql_dangereux):
    with pytest.raises(ValueError):
        validate_sql(sql_dangereux)
