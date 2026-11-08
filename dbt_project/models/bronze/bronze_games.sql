-- Bronze-tason taulu: Alkuperäinen otteludata sellaisenaan tallennettuna DuckDB:hen
{{ config(materialized='table', schema='bronze') }}

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
    raw_json,
    ingested_at
FROM {{ source('bronze', 'raw_games') }}
