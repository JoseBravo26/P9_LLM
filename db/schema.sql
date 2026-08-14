PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL UNIQUE,
    team_code TEXT NOT NULL,
    age INTEGER CHECK(age BETWEEN 15 AND 60),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date TEXT,
    home_team_code TEXT,
    away_team_code TEXT,
    home_score INTEGER,
    away_score INTEGER,
    source_file TEXT NOT NULL,
    extraction_confidence REAL NOT NULL DEFAULT 0 CHECK(extraction_confidence BETWEEN 0 AND 1),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_date, home_team_code, away_team_code, source_file)
);
CREATE TABLE IF NOT EXISTS stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    match_id INTEGER,
    season_label TEXT NOT NULL,
    granularity TEXT NOT NULL CHECK(granularity IN ('season','match')),
    games_played INTEGER CHECK(games_played >= 0),
    wins INTEGER CHECK(wins >= 0),
    losses INTEGER CHECK(losses >= 0),
    minutes_per_game REAL CHECK(minutes_per_game >= 0),
    total_points INTEGER CHECK(total_points >= 0),
    field_goal_pct REAL CHECK(field_goal_pct BETWEEN 0 AND 100),
    three_point_made INTEGER CHECK(three_point_made >= 0),
    three_point_attempted INTEGER CHECK(three_point_attempted >= 0),
    three_point_pct REAL CHECK(three_point_pct BETWEEN 0 AND 100),
    free_throw_pct REAL CHECK(free_throw_pct BETWEEN 0 AND 100),
    offensive_rebounds INTEGER CHECK(offensive_rebounds >= 0),
    defensive_rebounds INTEGER CHECK(defensive_rebounds >= 0),
    total_rebounds INTEGER CHECK(total_rebounds >= 0),
    assists INTEGER CHECK(assists >= 0),
    turnovers INTEGER CHECK(turnovers >= 0),
    steals INTEGER CHECK(steals >= 0),
    blocks INTEGER CHECK(blocks >= 0),
    offensive_rating REAL,
    defensive_rating REAL,
    net_rating REAL,
    true_shooting_pct REAL CHECK(true_shooting_pct BETWEEN 0 AND 100),
    usage_pct REAL CHECK(usage_pct BETWEEN 0 AND 100),
    source_file TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(player_id) REFERENCES players(player_id),
    FOREIGN KEY(match_id) REFERENCES matches(match_id),
    UNIQUE(player_id, match_id, season_label, granularity)
);
CREATE TABLE IF NOT EXISTS reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_file TEXT NOT NULL UNIQUE,
    page_count INTEGER CHECK(page_count >= 0),
    source_type TEXT NOT NULL DEFAULT 'pdf',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(match_id) REFERENCES matches(match_id)
);
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_code);
CREATE INDEX IF NOT EXISTS idx_stats_player ON stats(player_id);
CREATE INDEX IF NOT EXISTS idx_stats_granularity ON stats(granularity);
CREATE INDEX IF NOT EXISTS idx_reports_match ON reports(match_id);
