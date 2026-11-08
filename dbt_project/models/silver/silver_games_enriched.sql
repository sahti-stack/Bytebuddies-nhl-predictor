WITH games AS (
    SELECT * FROM {{ ref('stg_games') }}
),

team_metrics AS (
    SELECT * FROM {{ ref('silver_team_metrics') }}
),

-- Unpivot games so each game has one row for each team (for rolling calculations)
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
        -- Rest days from previous game in the same season
        date_diff(
            'day',
            LAG(game_date) OVER (
                PARTITION BY season, team_abbrev
                ORDER BY game_date, game_id
            ),
            game_date
        ) AS days_rest,
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
    -- Teams
    g.home_team_id,
    g.home_team_abbrev,
    g.home_team_name,
    g.away_team_id,
    g.away_team_abbrev,
    g.away_team_name,
    -- Actual outcomes (for targets)
    g.home_score,
    g.away_score,
    g.total_goals,
    g.goal_diff,
    g.home_win,
    g.away_win,
    g.is_regulation,
    g.is_overtime,
    -- Home team rest & form
    COALESCE(hf.days_rest, 5) AS home_rest_days,
    CASE WHEN hf.days_rest = 1 THEN 1 ELSE 0 END AS is_b2b_home,
    ROUND(hf.rolling_win_pct_5, 3) AS home_rolling_win_pct_5,
    ROUND(hf.rolling_goals_for_5, 2) AS home_rolling_goals_for_5,
    ROUND(hf.rolling_goals_against_5, 2) AS home_rolling_goals_against_5,
    ROUND(hf.rolling_win_pct_10, 3) AS home_rolling_win_pct_10,
    ROUND(hf.rolling_goals_for_10, 2) AS home_rolling_goals_for_10,
    ROUND(hf.rolling_goals_against_10, 2) AS home_rolling_goals_against_10,
    -- Away team rest & form
    COALESCE(af.days_rest, 5) AS away_rest_days,
    CASE WHEN af.days_rest = 1 THEN 1 ELSE 0 END AS is_b2b_away,
    ROUND(af.rolling_win_pct_5, 3) AS away_rolling_win_pct_5,
    ROUND(af.rolling_goals_for_5, 2) AS away_rolling_goals_for_5,
    ROUND(af.rolling_goals_against_5, 2) AS away_rolling_goals_against_5,
    ROUND(af.rolling_win_pct_10, 3) AS away_rolling_win_pct_10,
    ROUND(af.rolling_goals_for_10, 2) AS away_rolling_goals_for_10,
    ROUND(af.rolling_goals_against_10, 2) AS away_rolling_goals_against_10,
    -- Home team season metrics
    hm.point_pct AS home_point_pct,
    hm.goals_for_per_game AS home_goals_for_per_game,
    hm.goals_against_per_game AS home_goals_against_per_game,
    hm.power_play_pct AS home_power_play_pct,
    hm.penalty_kill_pct AS home_penalty_kill_pct,
    hm.shots_for_per_game AS home_shots_for_per_game,
    hm.shots_against_per_game AS home_shots_against_per_game,
    hm.faceoff_win_pct AS home_faceoff_win_pct,
    hm.corsi_pct AS home_corsi_pct,
    hm.fenwick_pct AS home_fenwick_pct,
    hm.pdo_5v5 AS home_pdo_5v5,
    -- Away team season metrics
    am.point_pct AS away_point_pct,
    am.goals_for_per_game AS away_goals_for_per_game,
    am.goals_against_per_game AS away_goals_against_per_game,
    am.power_play_pct AS away_power_play_pct,
    am.penalty_kill_pct AS away_penalty_kill_pct,
    am.shots_for_per_game AS away_shots_for_per_game,
    am.shots_against_per_game AS away_shots_against_per_game,
    am.faceoff_win_pct AS away_faceoff_win_pct,
    am.corsi_pct AS away_corsi_pct,
    am.fenwick_pct AS away_fenwick_pct,
    am.pdo_5v5 AS away_pdo_5v5
FROM games g
LEFT JOIN home_team_features hf
    ON g.game_id = hf.game_id
LEFT JOIN away_team_features af
    ON g.game_id = af.game_id
LEFT JOIN team_metrics hm
    ON g.season = hm.season
    AND (g.home_team_id = hm.team_id OR g.home_team_abbrev = hm.team_abbrev)
LEFT JOIN team_metrics am
    ON g.season = am.season
    AND (g.away_team_id = am.team_id OR g.away_team_abbrev = am.team_abbrev)
