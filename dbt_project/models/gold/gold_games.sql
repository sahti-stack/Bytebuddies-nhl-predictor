WITH enriched AS (
    SELECT * FROM {{ ref('silver_games_enriched') }}
)

SELECT
    -- =========================================================================
    -- GAME IDENTIFIERS & METADATA
    -- Match identification, scheduling timestamps, venue, and status
    -- =========================================================================
    game_id,
    season,
    game_type,
    game_date,
    start_time_utc,
    venue,
    game_state,
    is_played,

    -- =========================================================================
    -- TEAM IDENTIFIERS
    -- Official team IDs, 3-letter abbreviations, and franchise names
    -- =========================================================================
    home_team_id,
    home_team_abbrev,
    home_team_name,
    away_team_id,
    away_team_abbrev,
    away_team_name,

    -- =========================================================================
    -- OUTCOMES & PREDICTION TARGETS
    -- Realized goals, goal differentials, and winners (NULL for unplayed games)
    -- =========================================================================
    home_score,
    away_score,
    total_goals,
    goal_diff,
    home_win,
    away_win,
    is_regulation,
    is_overtime,

    -- =========================================================================
    -- SEASON PHASE & COLD-START SAMPLE SELECTION FLAGS
    -- Sequential game counts per team and boolean eligibility flags.
    -- Enables training/evaluating models either on the full season or strictly
    -- when sufficient in-season sample size (e.g. 5 or 10 games) is reached.
    -- =========================================================================
    home_game_num_season,
    away_game_num_season,
    CASE
        WHEN home_game_num_season >= 5 AND away_game_num_season >= 5 THEN 1
        ELSE 0
    END AS has_min_5_games,
    CASE
        WHEN home_game_num_season >= 10 AND away_game_num_season >= 10 THEN 1
        ELSE 0
    END AS has_min_10_games,

    -- =========================================================================
    -- SCHEDULE FATIGUE & REST METRICS
    -- Rest days, back-to-backs (B2B), 3-in-4 congestion, and 7-day game density
    -- =========================================================================
    -- Raw rest days since previous game (defaults to 5 for season openers)
    home_rest_days,
    away_rest_days,
    (home_rest_days - away_rest_days) AS rest_days_diff,

    -- Capped rest days (max 7 days) to prevent extreme outlier breaks from distorting linear models
    home_rest_days_capped,
    away_rest_days_capped,
    (home_rest_days_capped - away_rest_days_capped) AS rest_days_capped_diff,

    -- Back-to-back indicators (1 if game played on consecutive calendar days, 0 otherwise)
    is_b2b_home,
    is_b2b_away,
    (is_b2b_home - is_b2b_away) AS is_b2b_diff,

    -- Schedule congestion: 3rd game in 4 calendar days
    is_3_in_4_home,
    is_3_in_4_away,
    (is_3_in_4_home - is_3_in_4_away) AS is_3_in_4_diff,

    -- 7-day rolling game density and differential
    games_last_7_days_home,
    games_last_7_days_away,
    games_last_7_days_diff,

    -- Extreme schedule mismatch flags
    is_rested_vs_tired,
    is_tired_vs_rested,

    -- =========================================================================
    -- CUMULATIVE IN-SEASON PERFORMANCE PRIOR TO MATCH (EXPANDING WINDOW)
    -- Captures true current-season strength without look-ahead data leakage.
    -- Calculated over expanding window: ROWS UNBOUNDED PRECEDING AND 1 PRECEDING.
    -- =========================================================================
    home_in_season_games_played,
    away_in_season_games_played,
    home_in_season_win_pct,
    away_in_season_win_pct,
    ROUND(home_in_season_win_pct - away_in_season_win_pct, 3) AS in_season_win_pct_diff,
    home_in_season_goals_for_per_game,
    away_in_season_goals_for_per_game,
    ROUND(home_in_season_goals_for_per_game - away_in_season_goals_for_per_game, 2) AS in_season_goals_for_diff_per_game,
    home_in_season_goals_against_per_game,
    away_in_season_goals_against_per_game,
    ROUND(home_in_season_goals_against_per_game - away_in_season_goals_against_per_game, 2) AS in_season_goals_against_diff_per_game,
    ROUND(
        (home_in_season_goals_for_per_game - home_in_season_goals_against_per_game)
        - (away_in_season_goals_for_per_game - away_in_season_goals_against_per_game),
        2
    ) AS in_season_goal_diff_per_game_diff,

    -- =========================================================================
    -- SHORT-TERM RECENT FORM PRIOR TO MATCH (ROLLING 5 & 10 GAMES)
    -- Captures recent momentum and scoring trends without data leakage
    -- =========================================================================
    -- 5-game rolling form
    home_rolling_win_pct_5,
    away_rolling_win_pct_5,
    ROUND(home_rolling_win_pct_5 - away_rolling_win_pct_5, 3) AS win_pct_diff_5,

    home_rolling_goals_for_5,
    away_rolling_goals_for_5,
    ROUND(home_rolling_goals_for_5 - away_rolling_goals_for_5, 2) AS goals_for_diff_5,

    home_rolling_goals_against_5,
    away_rolling_goals_against_5,
    ROUND(home_rolling_goals_against_5 - away_rolling_goals_against_5, 2) AS goals_against_diff_5,

    -- 10-game rolling form
    home_rolling_win_pct_10,
    away_rolling_win_pct_10,
    ROUND(home_rolling_win_pct_10 - away_rolling_win_pct_10, 3) AS win_pct_diff_10,

    home_rolling_goals_for_10,
    away_rolling_goals_for_10,
    ROUND(home_rolling_goals_for_10 - away_rolling_goals_for_10, 2) AS goals_for_diff_10,

    home_rolling_goals_against_10,
    away_rolling_goals_against_10,
    ROUND(home_rolling_goals_against_10 - away_rolling_goals_against_10, 2) AS goals_against_diff_10,

    -- =========================================================================
    -- HISTORICAL BASELINE METRICS (PRIOR SEASON TEAM PERFORMANCE)
    -- Known prior to game 1 of the season. Essential for early-season predictions.
    -- =========================================================================
    home_prior_point_pct,
    away_prior_point_pct,
    ROUND(home_prior_point_pct - away_prior_point_pct, 4) AS prior_point_pct_diff,

    home_prior_goals_for_per_game,
    away_prior_goals_for_per_game,
    ROUND(home_prior_goals_for_per_game - away_prior_goals_for_per_game, 3) AS prior_goals_for_diff_per_game,

    home_prior_goals_against_per_game,
    away_prior_goals_against_per_game,
    ROUND(home_prior_goals_against_per_game - away_prior_goals_against_per_game, 3) AS prior_goals_against_diff_per_game,

    -- Special teams (Power Play & Penalty Kill)
    home_prior_power_play_pct,
    away_prior_power_play_pct,
    home_prior_penalty_kill_pct,
    away_prior_penalty_kill_pct,
    ROUND(
        (home_prior_power_play_pct + home_prior_penalty_kill_pct)
        - (away_prior_power_play_pct + away_prior_penalty_kill_pct),
        4
    ) AS prior_special_teams_diff,

    -- Shots and faceoffs
    home_prior_shots_for_per_game,
    away_prior_shots_for_per_game,
    ROUND(home_prior_shots_for_per_game - away_prior_shots_for_per_game, 2) AS prior_shots_diff_per_game,
    home_prior_shots_against_per_game,
    away_prior_shots_against_per_game,
    ROUND(home_prior_shots_against_per_game - away_prior_shots_against_per_game, 2) AS prior_shots_against_diff_per_game,
    home_prior_faceoff_win_pct,
    away_prior_faceoff_win_pct,
    ROUND(home_prior_faceoff_win_pct - away_prior_faceoff_win_pct, 4) AS prior_faceoff_diff,

    -- Advanced possession metrics (Corsi, Fenwick, PDO 5v5)
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
