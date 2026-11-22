-- Utah's 2024-25 prior-season baseline must equal Arizona's 2023-24 baseline.
-- Comparing against the source row (instead of hardcoded floats) keeps this
-- test valid if the NHL revises historical statistics.
WITH ari AS (
    SELECT
        point_pct,
        goals_for_per_game,
        goals_against_per_game,
        power_play_pct,
        penalty_kill_pct,
        shots_for_per_game,
        shots_against_per_game,
        faceoff_win_pct,
        corsi_pct,
        fenwick_pct,
        pdo_5v5
    FROM {{ ref('silver_team_metrics') }}
    WHERE season = 20232024
      AND team_id = 53
),

uta AS (
    SELECT DISTINCT
        home_prior_point_pct AS point_pct,
        home_prior_goals_for_per_game AS goals_for_per_game,
        home_prior_goals_against_per_game AS goals_against_per_game,
        home_prior_power_play_pct AS power_play_pct,
        home_prior_penalty_kill_pct AS penalty_kill_pct,
        home_prior_shots_for_per_game AS shots_for_per_game,
        home_prior_shots_against_per_game AS shots_against_per_game,
        home_prior_faceoff_win_pct AS faceoff_win_pct,
        home_prior_corsi_pct AS corsi_pct,
        home_prior_fenwick_pct AS fenwick_pct,
        home_prior_pdo_5v5 AS pdo_5v5
    FROM {{ ref('silver_games_enriched') }}
    WHERE season = 20242025
      AND home_team_id = 59

    UNION

    SELECT DISTINCT
        away_prior_point_pct AS point_pct,
        away_prior_goals_for_per_game AS goals_for_per_game,
        away_prior_goals_against_per_game AS goals_against_per_game,
        away_prior_power_play_pct AS power_play_pct,
        away_prior_penalty_kill_pct AS penalty_kill_pct,
        away_prior_shots_for_per_game AS shots_for_per_game,
        away_prior_shots_against_per_game AS shots_against_per_game,
        away_prior_faceoff_win_pct AS faceoff_win_pct,
        away_prior_corsi_pct AS corsi_pct,
        away_prior_fenwick_pct AS fenwick_pct,
        away_prior_pdo_5v5 AS pdo_5v5
    FROM {{ ref('silver_games_enriched') }}
    WHERE season = 20242025
      AND away_team_id = 59
)

SELECT
    (SELECT COUNT(*) FROM ari) AS ari_rows,
    (SELECT COUNT(*) FROM uta) AS uta_rows,
    (
        SELECT COUNT(*)
        FROM uta u
        CROSS JOIN ari a
        WHERE u.point_pct IS DISTINCT FROM a.point_pct
           OR u.goals_for_per_game IS DISTINCT FROM a.goals_for_per_game
           OR u.goals_against_per_game IS DISTINCT FROM a.goals_against_per_game
           OR u.power_play_pct IS DISTINCT FROM a.power_play_pct
           OR u.penalty_kill_pct IS DISTINCT FROM a.penalty_kill_pct
           OR u.shots_for_per_game IS DISTINCT FROM a.shots_for_per_game
           OR u.shots_against_per_game IS DISTINCT FROM a.shots_against_per_game
           OR u.faceoff_win_pct IS DISTINCT FROM a.faceoff_win_pct
           OR u.corsi_pct IS DISTINCT FROM a.corsi_pct
           OR u.fenwick_pct IS DISTINCT FROM a.fenwick_pct
           OR u.pdo_5v5 IS DISTINCT FROM a.pdo_5v5
    ) AS mismatch_rows
HAVING ari_rows <> 1
    OR uta_rows <> 1
    OR mismatch_rows > 0
