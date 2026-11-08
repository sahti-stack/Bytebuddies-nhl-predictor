-- Bronze-tason taulu: Alkuperäiset kehittyneet Corsi-, Fenwick- ja PDO-tilastot
{{ config(materialized='table', schema='bronze') }}

SELECT
    season,
    team_id,
    team_abbrev,
    sat_pct,
    usat_pct,
    shooting_plus_save_pct_5v5,
    raw_json,
    ingested_at
FROM {{ source('bronze', 'raw_team_percentages') }}
