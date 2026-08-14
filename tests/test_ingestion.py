"""Tests de validation Pydantic appliques avant toute insertion en base."""
import pytest
from pydantic import ValidationError

from utils.db_schemas import SeasonStatRow


def test_ligne_saison_valide_est_acceptee():
    row = SeasonStatRow(
        player_name="Test Player", team_code="TST", age=25, season_label="2024-2025",
        games_played=10, wins=6, losses=4, three_point_made=5, three_point_attempted=10,
    )
    assert row.team_code == "TST"
    assert row.three_point_made <= row.three_point_attempted


def test_tirs_reussis_superieurs_aux_tentatives_est_refuse():
    with pytest.raises(ValidationError):
        SeasonStatRow(
            player_name="Incoherent Player", team_code="TST", season_label="2024-2025",
            games_played=10, wins=5, losses=5, three_point_made=20, three_point_attempted=10,
        )


def test_age_hors_bornes_est_refuse():
    with pytest.raises(ValidationError):
        SeasonStatRow(
            player_name="Trop Jeune", team_code="TST", age=10, season_label="2024-2025",
            games_played=1, wins=0, losses=1,
        )
