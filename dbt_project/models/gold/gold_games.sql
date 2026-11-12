WITH enriched AS (
    SELECT * FROM {{ ref('silver_games_enriched') }}
)

SELECT
    -- Ottelun perustiedot
    game_id,
    season,
    game_type,
    game_date,
    start_time_utc,
    venue,
    game_state,
    is_played,

    -- Joukkuetiedot
    home_team_id,
    home_team_abbrev,
    home_team_name,
    away_team_id,
    away_team_abbrev,
    away_team_name,

    -- Ottelun lopputulos ja mahdolliset ennustekohteet (Targets)
    home_score,
    away_score,
    total_goals,
    goal_diff,
    home_win,
    away_win,
    is_regulation,
    is_overtime,

    -- Lepo- ja rasituspiirteet
    home_rest_days,
    away_rest_days,
    (home_rest_days - away_rest_days) AS rest_days_diff,
    is_b2b_home,
    is_b2b_away,

    -- Kuntopuntari ennen ottelua (Rolling 5 ja 10 ottelua ilman datavuotoa)
    home_rolling_win_pct_5,
    away_rolling_win_pct_5,
    ROUND(home_rolling_win_pct_5 - away_rolling_win_pct_5, 3) AS win_pct_diff_5,

    home_rolling_goals_for_5,
    away_rolling_goals_for_5,
    ROUND(home_rolling_goals_for_5 - away_rolling_goals_for_5, 2) AS goals_for_diff_5,

    home_rolling_goals_against_5,
    away_rolling_goals_against_5,
    ROUND(home_rolling_goals_against_5 - away_rolling_goals_against_5, 2) AS goals_against_diff_5,

    home_rolling_win_pct_10,
    away_rolling_win_pct_10,
    ROUND(home_rolling_win_pct_10 - away_rolling_win_pct_10, 3) AS win_pct_diff_10,

    -- Joukkueiden edellisen kauden perustasotilastot
    home_prior_point_pct,
    away_prior_point_pct,
    ROUND(home_prior_point_pct - away_prior_point_pct, 4) AS prior_point_pct_diff,

    home_prior_goals_for_per_game,
    away_prior_goals_for_per_game,
    ROUND(home_prior_goals_for_per_game - away_prior_goals_for_per_game, 3) AS prior_goals_for_diff_per_game,

    home_prior_goals_against_per_game,
    away_prior_goals_against_per_game,
    ROUND(home_prior_goals_against_per_game - away_prior_goals_against_per_game, 3) AS prior_goals_against_diff_per_game,

    -- Erikoistilanteet edelliseltä kaudelta (Ylivoima & Alivoima)
    home_prior_power_play_pct,
    away_prior_power_play_pct,
    home_prior_penalty_kill_pct,
    away_prior_penalty_kill_pct,
    ROUND(
        (home_prior_power_play_pct + home_prior_penalty_kill_pct)
        - (away_prior_power_play_pct + away_prior_penalty_kill_pct),
        4
    ) AS prior_special_teams_diff,

    -- Laukaukset ja aloitukset edelliseltä kaudelta
    home_prior_shots_for_per_game,
    away_prior_shots_for_per_game,
    ROUND(home_prior_shots_for_per_game - away_prior_shots_for_per_game, 2) AS prior_shots_diff_per_game,
    ROUND(home_prior_faceoff_win_pct - away_prior_faceoff_win_pct, 4) AS prior_faceoff_diff,

    -- Kehittyneet tilastot edelliseltä kaudelta (Corsi, Fenwick, PDO)
    home_prior_corsi_pct,
    away_prior_corsi_pct,
    ROUND(home_prior_corsi_pct - away_prior_corsi_pct, 4) AS prior_corsi_diff,
    home_prior_fenwick_pct,
    away_prior_fenwick_pct,
    ROUND(home_prior_fenwick_pct - away_prior_fenwick_pct, 4) AS prior_fenwick_diff,
    home_prior_pdo_5v5,
    away_prior_pdo_5v5,
    ROUND(home_prior_pdo_5v5 - away_prior_pdo_5v5, 4) AS prior_pdo_diff
FROM enriched
