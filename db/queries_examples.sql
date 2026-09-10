-- Top 10 des meilleurs pourcentages a 3 points (>=100 tentatives)
SELECT p.player_name, p.team_code, s.three_point_pct, s.three_point_attempted
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.granularity = 'season' AND s.three_point_attempted >= 100
ORDER BY s.three_point_pct DESC LIMIT 10;

-- Comparaison de rebonds entre trois joueurs
SELECT p.player_name, s.total_rebounds
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.granularity = 'season'
  AND p.player_name IN ('Nikola Jokic','Karl-Anthony Towns','Giannis Antetokounmpo')
ORDER BY s.total_rebounds DESC;

-- Joueurs a plus de 20 points, 5 rebonds et 5 passes par match
SELECT p.player_name,
       s.total_points * 1.0 / s.games_played AS points_per_game,
       s.total_rebounds * 1.0 / s.games_played AS rebounds_per_game,
       s.assists * 1.0 / s.games_played AS assists_per_game
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.granularity = 'season' AND s.games_played > 0
  AND s.total_points * 1.0 / s.games_played >= 20
  AND s.total_rebounds * 1.0 / s.games_played >= 5
  AND s.assists * 1.0 / s.games_played >= 5
ORDER BY points_per_game DESC LIMIT 20;

-- Net rating moyen par equipe
SELECT p.team_code, AVG(s.net_rating) AS avg_net_rating
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.granularity = 'season'
GROUP BY p.team_code ORDER BY avg_net_rating DESC;

-- Rapports lies a un match identifie
SELECT r.title, r.source_file, m.match_date, m.home_team_code, m.away_team_code
FROM reports r JOIN matches m ON m.match_id = r.match_id;
