"""Exemples question -> SQL fournis au LLM pour cadrer la generation SQLite.

Chaque exemple illustre soit une requete correcte sur les statistiques de saison,
soit le motif d abstention attendu quand la granularite demandee est indisponible
(cinq derniers matchs, domicile/exterieur).
"""

FEW_SHOT_SQL = [
    {
        "question": "Quels sont les dix meilleurs pourcentages a trois points avec au moins 100 tentatives ?",
        "sql": (
            "SELECT p.player_name, p.team_code, s.three_point_pct, s.three_point_attempted "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = \'season\' AND s.three_point_attempted >= 100 "
            "ORDER BY s.three_point_pct DESC LIMIT 10"
        ),
    },
    {
        "question": "Compare les rebonds de Nikola Jokic, Karl-Anthony Towns et Giannis Antetokounmpo.",
        "sql": (
            "SELECT p.player_name, s.total_rebounds FROM stats s "
            "JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = \'season\' AND p.player_name IN "
            "(\'Nikola Jokic\', \'Karl-Anthony Towns\', \'Giannis Antetokounmpo\') "
            "ORDER BY s.total_rebounds DESC"
        ),
    },
    {
        "question": "Quels joueurs depassent 20 points, 5 rebonds et 5 passes par match ?",
        "sql": (
            "SELECT p.player_name, "
            "s.total_points * 1.0 / s.games_played AS points_per_game, "
            "s.total_rebounds * 1.0 / s.games_played AS rebounds_per_game, "
            "s.assists * 1.0 / s.games_played AS assists_per_game "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = \'season\' AND s.games_played > 0 "
            "AND s.total_points * 1.0 / s.games_played >= 20 "
            "AND s.total_rebounds * 1.0 / s.games_played >= 5 "
            "AND s.assists * 1.0 / s.games_played >= 5 "
            "ORDER BY points_per_game DESC LIMIT 20"
        ),
    },
    {
        "question": "Compare les statistiques de rebonds de l equipe a domicile et a l exterieur.",
        "sql": "ABSTAIN",
    },
    {
        "question": "Quel joueur a le meilleur pourcentage a trois points sur les cinq derniers matchs ?",
        "sql": "ABSTAIN",
    },
]
