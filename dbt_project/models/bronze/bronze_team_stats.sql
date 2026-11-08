-- Bronze-tason taulu: Alkuperäiset joukkuekohtaiset kausitilastot
{{ config(materialized='table', schema='bronze') }}

SELECT
    season,
    team_id,
    team_abbrev,
    team_name,
    games_played,
    wins,
    losses,
    ot_losses,
    points,
    point_pct,
    goals_for_per_game,
    goals_against_per_game,
    power_play_pct,
    penalty_kill_pct,
    shots_for_per_game,
    shots_against_per_game,
    faceoff_win_pct,
    raw_json,
    ingested_at
FROM {{ source('bronze', 'raw_team_stats') }}
