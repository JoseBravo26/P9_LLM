"""Exemples question -> SQL fournis au LLM pour cadrer la generation SQLite.

Chaque exemple illustre soit une requete correcte sur les statistiques de saison,
soit le motif d abstention attendu quand la granularite demandee est indisponible
(cinq derniers matchs, domicile/exterieur).

Correctif du 18/08/2026 :
    Tous les exemples utilisant un nom de joueur ont ete reecrits pour
    utiliser `LIKE '%fragment%'` au lieu de l'egalite stricte `=`. En effet,
    les noms de joueurs stockes en base contiennent des accents (ex: "Nikola
    Jokić" avec l'accent croate sur le c), alors que les questions des
    utilisateurs, tout comme les anciens exemples de ce fichier, les
    ecrivaient sans accent ("Nikola Jokic"). Le LLM reproduisait fidelement
    l'orthographe sans accent des exemples dans le SQL genere, ce qui
    provoquait une comparaison stricte ne matchant aucune ligne (0 resultat),
    alors que la donnee existait bel et bien en base.
    Un nouvel exemple dedie a ce cas precis (pourcentage a 3 points de Jokic)
    a egalement ete ajoute pour ancrer le bon reflexe chez le LLM.
"""

FEW_SHOT_SQL = [
    {
        "question": "Quels sont les dix meilleurs pourcentages a trois points avec au moins 100 tentatives ?",
        "sql": (
            "SELECT p.player_name, p.team_code, s.three_point_pct, s.three_point_attempted "
            "FROM stats s JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND s.three_point_attempted >= 100 "
            "ORDER BY s.three_point_pct DESC LIMIT 10"
        ),
    },
    {
        "question": "Quel est le pourcentage a 3 points de Nikola Jokic ?",
        "sql": (
            "SELECT p.player_name, s.three_point_pct FROM stats s "
            "JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND p.player_name LIKE '%Jokic%' "
            "ORDER BY s.season_label DESC LIMIT 1"
        ),
    },
    {
        "question": "Compare les rebonds de Nikola Jokic, Karl-Anthony Towns et Giannis Antetokounmpo.",
        "sql": (
            "SELECT p.player_name, s.total_rebounds FROM stats s "
            "JOIN players p ON p.player_id = s.player_id "
            "WHERE s.granularity = 'season' AND ("
            "p.player_name LIKE '%Jokic%' OR p.player_name LIKE '%Towns%' "
            "OR p.player_name LIKE '%Antetokounmpo%') "
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
            "WHERE s.granularity = 'season' AND s.games_played > 0 "
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
