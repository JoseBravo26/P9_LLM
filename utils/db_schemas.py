"""Contrats Pydantic appliques avant toute insertion en base (players/stats)."""
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PlayerRow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    player_name: str = Field(min_length=2)
    team_code: str = Field(min_length=2, max_length=5)
    age: int | None = Field(default=None, ge=15, le=60)

    @field_validator("player_name", "team_code", mode="before")
    @classmethod
    def normaliser_texte(cls, valeur):
        return str(valeur).strip()


class SeasonStatRow(PlayerRow):
    """Ligne de statistiques saisonnieres, granularite = saison entiere."""
    season_label: str = Field(min_length=4)
    games_played: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    minutes_per_game: float | None = Field(default=None, ge=0)
    total_points: int | None = Field(default=None, ge=0)
    field_goal_pct: float | None = Field(default=None, ge=0, le=100)
    three_point_made: int | None = Field(default=None, ge=0)
    three_point_attempted: int | None = Field(default=None, ge=0)
    three_point_pct: float | None = Field(default=None, ge=0, le=100)
    free_throw_pct: float | None = Field(default=None, ge=0, le=100)
    offensive_rebounds: int | None = Field(default=None, ge=0)
    defensive_rebounds: int | None = Field(default=None, ge=0)
    total_rebounds: int | None = Field(default=None, ge=0)
    assists: int | None = Field(default=None, ge=0)
    turnovers: int | None = Field(default=None, ge=0)
    steals: int | None = Field(default=None, ge=0)
    blocks: int | None = Field(default=None, ge=0)
    offensive_rating: float | None = None
    defensive_rating: float | None = None
    net_rating: float | None = None
    true_shooting_pct: float | None = Field(default=None, ge=0, le=150)
    usage_pct: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def verifier_coherence_tirs(self):
        if (
            self.three_point_made is not None
            and self.three_point_attempted is not None
            and self.three_point_made > self.three_point_attempted
        ):
            raise ValueError("Le nombre de tirs a 3 points reussis depasse les tentatives.")
        return self


class SQLRequest(BaseModel):
    """Requete SQL generee par le LLM, avant execution controlee."""
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=1000)
    sql: str = Field(min_length=8, max_length=10_000)


class SQLToolResult(BaseModel):
    """Resultat structure renvoye par le SQL Tool a l'agent."""
    sql: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)
    row_count: int = 0
    error: str | None = None
