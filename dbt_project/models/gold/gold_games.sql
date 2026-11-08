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

    -- Joukkueiden kausitilastot
    home_point_pct,
    away_point_pct,
    ROUND(home_point_pct - away_point_pct, 4) AS point_pct_diff,

    home_goals_for_per_game,
    away_goals_for_per_game,
    ROUND(home_goals_for_per_game - away_goals_for_per_game, 3) AS goals_for_diff_per_game,

    home_goals_against_per_game,
    away_goals_against_per_game,
    ROUND(home_goals_against_per_game - away_goals_against_per_game, 3) AS goals_against_diff_per_game,

    -- Erikoistilanteet (Ylivoima & Alivoima)
    home_power_play_pct,
    away_power_play_pct,
    home_penalty_kill_pct,
    away_penalty_kill_pct,
    ROUND(
        (home_power_play_pct + home_penalty_kill_pct) - (away_power_play_pct + away_penalty_kill_pct),
        4
    ) AS special_teams_diff,

    -- Laukaukset ja aloitukset
    home_shots_for_per_game,
    away_shots_for_per_game,
    ROUND(home_shots_for_per_game - away_shots_for_per_game, 2) AS shots_diff_per_game,
    ROUND(home_faceoff_win_pct - away_faceoff_win_pct, 4) AS faceoff_diff,

    -- Kehittyneet tilastot (Corsi, Fenwick, PDO)
    home_corsi_pct,
    away_corsi_pct,
    ROUND(home_corsi_pct - away_corsi_pct, 4) AS corsi_diff,
    home_fenwick_pct,
    away_fenwick_pct,
    ROUND(home_fenwick_pct - away_fenwick_pct, 4) AS fenwick_diff,
    home_pdo_5v5,
    away_pdo_5v5,
    ROUND(home_pdo_5v5 - away_pdo_5v5, 4) AS pdo_diff
FROM enriched
