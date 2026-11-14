WITH games AS (
    -- Exclude preseason games (game_type = 1) at the silver layer.
    -- Include only competitive regular season (2) and playoff (3) games,
    -- ensuring models train and predict solely on meaningful official competition.
    SELECT * FROM {{ ref('stg_games') }}
    WHERE game_type IN (2, 3)
),

team_metrics AS (
    -- Consolidated historical team statistics and advanced metrics
    SELECT * FROM {{ ref('silver_team_metrics') }}
),

-- Unpivot games into two rows per match (home and away perspective)
-- Enables precise calculation of team-level rolling and window functions
team_game_events AS (
    -- Home team perspective
    SELECT
        game_id,
        season,
        game_date,
        game_type,
        home_team_abbrev AS team_abbrev,
        away_team_abbrev AS opponent_abbrev,
        TRUE AS is_home,
        home_score AS goals_for,
        away_score AS goals_against,
        home_win AS is_win,
        is_played
    FROM games

    UNION ALL

    -- Away team perspective
    SELECT
        game_id,
        season,
        game_date,
        game_type,
        away_team_abbrev AS team_abbrev,
        home_team_abbrev AS opponent_abbrev,
        FALSE AS is_home,
        away_score AS goals_for,
        home_score AS goals_against,
        away_win AS is_win,
        is_played
    FROM games
),

team_game_features AS (
    SELECT
        game_id,
        team_abbrev,
        is_home,
        game_date,

        -- Team's sequential game number in the current season (used for cold-start handling)
        ROW_NUMBER() OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
        ) AS team_game_number_season,

        -- Days of rest since previous game in the same season
        date_diff(
            'day',
            LAG(game_date) OVER (
                PARTITION BY season, team_abbrev
                ORDER BY game_date, game_id
            ),
            game_date
        ) AS days_rest,

        -- Flag: 3rd game in 4 calendar days (schedule congestion fatigue)
        CASE
            WHEN date_diff(
                'day',
                LAG(game_date, 2) OVER (
                    PARTITION BY season, team_abbrev
                    ORDER BY game_date, game_id
                ),
                game_date
            ) <= 3 THEN 1
            ELSE 0
        END AS is_3_in_4,

        -- Total games scheduled in the preceding 7 calendar days before this match
        (
            CASE WHEN date_diff('day', LAG(game_date, 1) OVER (PARTITION BY season, team_abbrev ORDER BY game_date, game_id), game_date) <= 7 THEN 1 ELSE 0 END +
            CASE WHEN date_diff('day', LAG(game_date, 2) OVER (PARTITION BY season, team_abbrev ORDER BY game_date, game_id), game_date) <= 7 THEN 1 ELSE 0 END +
            CASE WHEN date_diff('day', LAG(game_date, 3) OVER (PARTITION BY season, team_abbrev ORDER BY game_date, game_id), game_date) <= 7 THEN 1 ELSE 0 END +
            CASE WHEN date_diff('day', LAG(game_date, 4) OVER (PARTITION BY season, team_abbrev ORDER BY game_date, game_id), game_date) <= 7 THEN 1 ELSE 0 END +
            CASE WHEN date_diff('day', LAG(game_date, 5) OVER (PARTITION BY season, team_abbrev ORDER BY game_date, game_id), game_date) <= 7 THEN 1 ELSE 0 END
        ) AS games_last_7_days,

        -- Expanding cumulative in-season performance prior to this match (no data leakage)
        COUNT(CASE WHEN is_played THEN 1 END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS in_season_games_played,

        AVG(CASE WHEN is_played THEN is_win END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS in_season_win_pct,

        AVG(CASE WHEN is_played THEN goals_for END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS in_season_goals_for_per_game,

        AVG(CASE WHEN is_played THEN goals_against END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS in_season_goals_against_per_game,

        -- Rolling 5 games prior to this match (only completed games)
        AVG(CASE WHEN is_played THEN is_win END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS rolling_win_pct_5,
        AVG(CASE WHEN is_played THEN goals_for END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS rolling_goals_for_5,
        AVG(CASE WHEN is_played THEN goals_against END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS rolling_goals_against_5,

        -- Rolling 10 games prior to this match
        AVG(CASE WHEN is_played THEN is_win END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS rolling_win_pct_10,
        AVG(CASE WHEN is_played THEN goals_for END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS rolling_goals_for_10,
        AVG(CASE WHEN is_played THEN goals_against END) OVER (
            PARTITION BY season, team_abbrev
            ORDER BY game_date, game_id
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS rolling_goals_against_10
    FROM team_game_events
),

home_team_features AS (
    SELECT * FROM team_game_features WHERE is_home = TRUE
),

away_team_features AS (
    SELECT * FROM team_game_features WHERE is_home = FALSE
)

SELECT
    g.game_id,
    g.season,
    g.game_type,
    g.game_date,
    g.start_time_utc,
    g.venue,
    g.game_state,
    g.is_played,

    -- Team identifiers
    g.home_team_id,
    g.home_team_abbrev,
    g.home_team_name,
    g.away_team_id,
    g.away_team_abbrev,
    g.away_team_name,

    -- Match outcome targets for ML models
    g.home_score,
    g.away_score,
    g.total_goals,
    g.goal_diff,
    g.home_win,
    g.away_win,
    g.is_regulation,
    g.is_overtime,

    -- Sequential season game numbers for cold-start identification
    hf.team_game_number_season AS home_game_num_season,
    af.team_game_number_season AS away_game_num_season,

    -- Home team rest & fatigue metrics
    COALESCE(hf.days_rest, 5) AS home_rest_days,
    LEAST(COALESCE(hf.days_rest, 5), 7) AS home_rest_days_capped,
    CASE WHEN hf.days_rest = 1 THEN 1 ELSE 0 END AS is_b2b_home,
    hf.is_3_in_4 AS is_3_in_4_home,
    hf.games_last_7_days AS games_last_7_days_home,

    -- Away team rest & fatigue metrics
    COALESCE(af.days_rest, 5) AS away_rest_days,
    LEAST(COALESCE(af.days_rest, 5), 7) AS away_rest_days_capped,
    CASE WHEN af.days_rest = 1 THEN 1 ELSE 0 END AS is_b2b_away,
    af.is_3_in_4 AS is_3_in_4_away,
    af.games_last_7_days AS games_last_7_days_away,

    -- Schedule rest and congestion advantage differentials
    (COALESCE(hf.days_rest, 5) - COALESCE(af.days_rest, 5)) AS rest_days_diff,
    (hf.games_last_7_days - af.games_last_7_days) AS games_last_7_days_diff,
    -- Home rested (>= 2 days) vs. Away playing on back-to-back (1 day rest)
    CASE WHEN COALESCE(hf.days_rest, 5) >= 2 AND af.days_rest = 1 THEN 1 ELSE 0 END AS is_rested_vs_tired,
    -- Home on back-to-back vs. Away rested (>= 2 days)
    CASE WHEN hf.days_rest = 1 AND COALESCE(af.days_rest, 5) >= 2 THEN 1 ELSE 0 END AS is_tired_vs_rested,

    -- Cumulative in-season performance prior to this match (expanding window)
    COALESCE(hf.in_season_games_played, 0) AS home_in_season_games_played,
    COALESCE(af.in_season_games_played, 0) AS away_in_season_games_played,
    ROUND(hf.in_season_win_pct, 3) AS home_in_season_win_pct,
    ROUND(af.in_season_win_pct, 3) AS away_in_season_win_pct,
    ROUND(hf.in_season_goals_for_per_game, 2) AS home_in_season_goals_for_per_game,
    ROUND(af.in_season_goals_for_per_game, 2) AS away_in_season_goals_for_per_game,
    ROUND(hf.in_season_goals_against_per_game, 2) AS home_in_season_goals_against_per_game,
    ROUND(af.in_season_goals_against_per_game, 2) AS away_in_season_goals_against_per_game,

    -- Home team rolling 5 and 10 game recent form
    ROUND(hf.rolling_win_pct_5, 3) AS home_rolling_win_pct_5,
    ROUND(hf.rolling_goals_for_5, 2) AS home_rolling_goals_for_5,
    ROUND(hf.rolling_goals_against_5, 2) AS home_rolling_goals_against_5,
    ROUND(hf.rolling_win_pct_10, 3) AS home_rolling_win_pct_10,
    ROUND(hf.rolling_goals_for_10, 2) AS home_rolling_goals_for_10,
    ROUND(hf.rolling_goals_against_10, 2) AS home_rolling_goals_against_10,

    -- Away team rolling 5 and 10 game recent form
    ROUND(af.rolling_win_pct_5, 3) AS away_rolling_win_pct_5,
    ROUND(af.rolling_goals_for_5, 2) AS away_rolling_goals_for_5,
    ROUND(af.rolling_goals_against_5, 2) AS away_rolling_goals_against_5,
    ROUND(af.rolling_win_pct_10, 3) AS away_rolling_win_pct_10,
    ROUND(af.rolling_goals_for_10, 2) AS away_rolling_goals_for_10,
    ROUND(af.rolling_goals_against_10, 2) AS away_rolling_goals_against_10,

    -- Home team prior-season baseline metrics (prevents look-ahead data leakage)
    hm.point_pct AS home_prior_point_pct,
    hm.goals_for_per_game AS home_prior_goals_for_per_game,
    hm.goals_against_per_game AS home_prior_goals_against_per_game,
    hm.power_play_pct AS home_prior_power_play_pct,
    hm.penalty_kill_pct AS home_prior_penalty_kill_pct,
    hm.shots_for_per_game AS home_prior_shots_for_per_game,
    hm.shots_against_per_game AS home_prior_shots_against_per_game,
    hm.faceoff_win_pct AS home_prior_faceoff_win_pct,
    hm.corsi_pct AS home_prior_corsi_pct,
    hm.fenwick_pct AS home_prior_fenwick_pct,
    hm.pdo_5v5 AS home_prior_pdo_5v5,

    -- Away team prior-season baseline metrics (prevents look-ahead data leakage)
    am.point_pct AS away_prior_point_pct,
    am.goals_for_per_game AS away_prior_goals_for_per_game,
    am.goals_against_per_game AS away_prior_goals_against_per_game,
    am.power_play_pct AS away_prior_power_play_pct,
    am.penalty_kill_pct AS away_prior_penalty_kill_pct,
    am.shots_for_per_game AS away_prior_shots_for_per_game,
    am.shots_against_per_game AS away_prior_shots_against_per_game,
    am.faceoff_win_pct AS away_prior_faceoff_win_pct,
    am.corsi_pct AS away_prior_corsi_pct,
    am.fenwick_pct AS away_prior_fenwick_pct,
    am.pdo_5v5 AS away_prior_pdo_5v5
FROM games g
LEFT JOIN home_team_features hf
    ON g.game_id = hf.game_id
LEFT JOIN away_team_features af
    ON g.game_id = af.game_id
LEFT JOIN team_metrics hm
    ON (g.season - 10001) = hm.season
    AND (g.home_team_id = hm.team_id OR g.home_team_abbrev = hm.team_abbrev)
LEFT JOIN team_metrics am
    ON (g.season - 10001) = am.season
    AND (g.away_team_id = am.team_id OR g.away_team_abbrev = am.team_abbrev)
