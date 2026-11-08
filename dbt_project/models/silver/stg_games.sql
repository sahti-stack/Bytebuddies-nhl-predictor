WITH source_games AS (
    SELECT * FROM {{ ref('bronze_games') }}
),

ranked AS (
    SELECT
        game_id,
        season,
        game_type,
        TRY_CAST(game_date AS DATE) AS game_date,
        start_time_utc,
        venue,
        game_state,
        home_team_id,
        home_team_abbrev,
        home_team_name,
        home_score,
        away_team_id,
        away_team_abbrev,
        away_team_name,
        away_score,
        outcome_last_period_type,
        ingested_at,
        ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY ingested_at DESC) AS rn
    FROM source_games
)

SELECT
    game_id,
    season,
    game_type,
    game_date,
    start_time_utc,
    venue,
    game_state,
    home_team_id,
    home_team_abbrev,
    home_team_name,
    home_score,
    away_team_id,
    away_team_abbrev,
    away_team_name,
    away_score,
    outcome_last_period_type,
    CASE
        WHEN game_state IN ('FINAL', 'OFF') AND home_score IS NOT NULL AND away_score IS NOT NULL
        THEN TRUE
        ELSE FALSE
    END AS is_played,
    CASE
        WHEN game_state IN ('FINAL', 'OFF') AND home_score > away_score THEN 1
        WHEN game_state IN ('FINAL', 'OFF') AND home_score < away_score THEN 0
        ELSE NULL
    END AS home_win,
    CASE
        WHEN game_state IN ('FINAL', 'OFF') AND away_score > home_score THEN 1
        WHEN game_state IN ('FINAL', 'OFF') AND away_score < home_score THEN 0
        ELSE NULL
    END AS away_win,
    CASE
        WHEN game_state IN ('FINAL', 'OFF')
        THEN home_score + away_score
        ELSE NULL
    END AS total_goals,
    CASE
        WHEN game_state IN ('FINAL', 'OFF')
        THEN home_score - away_score
        ELSE NULL
    END AS goal_diff,
    CASE
        WHEN outcome_last_period_type = 'REG' THEN TRUE
        WHEN outcome_last_period_type IN ('OT', 'SO') THEN FALSE
        ELSE NULL
    END AS is_regulation,
    CASE
        WHEN outcome_last_period_type IN ('OT', 'SO') THEN TRUE
        WHEN outcome_last_period_type = 'REG' THEN FALSE
        ELSE NULL
    END AS is_overtime,
    ingested_at
FROM ranked
WHERE rn = 1
