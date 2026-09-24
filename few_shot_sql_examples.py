"""Exemples question -> SQL pour guider la génération SQLite.

Les exemples couvrent les statistiques saisonnières, les comparaisons de
joueurs, les formulations bruitées et les demandes qui doivent conduire à
une abstention. Les noms des joueurs sont écrits sans accent dans les motifs
LIKE : le SQL Tool applique ensuite sa normalisation SANS_ACCENTS.
"""

FEW_SHOT_SQL = [
    {
        "question": "Quels sont les dix meilleurs pourcentages à trois points avec au moins 100 tentatives ?",
        "sql": (
            "SELECT p.player_name, p.team_code, s.three_point_pct, s.three_point_attempted "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND s.three_point_attempted >= 100 "
            "ORDER BY s.three_point_pct DESC LIMIT 10"
        ),
    },
    {
        "question": "Quel est le pourcentage à 3 points de Nikola Jokić ?",
        "sql": (
            "SELECT p.player_name, s.three_point_pct "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND p.player_name LIKE '%Jokic%' "
            "ORDER BY s.season_label DESC LIMIT 1"
        ),
    },
    {
        "question": "Compare les rebonds de Nikola Jokić, Karl-Anthony Towns et Giannis Antetokounmpo.",
        "sql": (
            "SELECT p.player_name, s.total_rebounds "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND ("
            "p.player_name LIKE '%Jokic%' OR p.player_name LIKE '%Towns%' "
            "OR p.player_name LIKE '%Antetokounmpo%') "
            "ORDER BY s.total_rebounds DESC LIMIT 3"
        ),
    },
    {
        "question": "Quels joueurs dépassent 20 points, 5 rebonds et 5 passes par match ?",
        "sql": (
            "SELECT p.player_name, "
            "s.total_points * 1.0 / s.games_played AS points_per_game, "
            "s.total_rebounds * 1.0 / s.games_played AS rebounds_per_game, "
            "s.assists * 1.0 / s.games_played AS assists_per_game "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND s.games_played > 0 "
            "AND s.total_points * 1.0 / s.games_played >= 20 "
            "AND s.total_rebounds * 1.0 / s.games_played >= 5 "
            "AND s.assists * 1.0 / s.games_played >= 5 "
            "ORDER BY points_per_game DESC LIMIT 20"
        ),
    },
    {
        "question": "Compare les statistiques de rebonds de l'équipe à domicile et à l'extérieur.",
        "sql": "ABSTAIN",
    },
    {
        "question": "Quel joueur a le meilleur pourcentage à trois points sur les cinq derniers matchs ?",
        "sql": "ABSTAIN",
    },
    {
        "question": "Quel était le pourcentage à 3 points de Stephen Curry lors de son dernier match ?",
        "sql": "ABSTAIN",
    },
    {
        "question": "Compare les rebonds de Nikola Jokić en playoffs et en saison régulière.",
        "sql": "ABSTAIN",
    },
    {
        "question": "Combien de rebonds Karl-Anthony Towns a-t-il pris contre les Celtics ?",
        "sql": "ABSTAIN",
    },
    {
        "question": "Quel est le salaire de Nikola Jokić cette saison ?",
        "sql": "ABSTAIN",
    },
    {
        "question": "Quelle est la différence de pourcentage à 3 points entre Stephen Curry et Anthony Edwards ?",
        "sql": (
            "SELECT p.player_name, s.three_point_pct "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND ("
            "p.player_name LIKE '%Stephen Curry%' "
            "OR p.player_name LIKE '%Anthony Edwards%') "
            "ORDER BY p.player_name ASC LIMIT 2"
        ),
    },
    {
        "question": "jokic a mi cb de pts ?",
        "sql": (
            "SELECT p.player_name, s.total_points "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND p.player_name LIKE '%Jokic%' "
            "LIMIT 1"
        ),
    },
    {
        "question": "giannis fg% stp",
        "sql": (
            "SELECT p.player_name, s.field_goal_pct "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND p.player_name LIKE '%Giannis%' "
            "LIMIT 1"
        ),
    },
    {
        "question": "c lekel le + fort au shoot entre booker et tatum ?",
        "sql": (
            "SELECT p.player_name, s.field_goal_pct "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND ("
            "p.player_name LIKE '%Booker%' OR p.player_name LIKE '%Tatum%') "
            "ORDER BY s.field_goal_pct DESC LIMIT 2"
        ),
    },
]
