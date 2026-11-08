WITH source_percentages AS (
    SELECT * FROM {{ ref('bronze_team_percentages') }}
),

ranked AS (
    SELECT
        season,
        team_id,
        team_abbrev,
        sat_pct,
        usat_pct,
        shooting_plus_save_pct_5v5,
        ingested_at,
        ROW_NUMBER() OVER (PARTITION BY season, team_id ORDER BY ingested_at DESC) AS rn
    FROM source_percentages
)

SELECT
    season,
    team_id,
    team_abbrev,
    ROUND(sat_pct, 4) AS sat_pct,
    ROUND(usat_pct, 4) AS usat_pct,
    ROUND(shooting_plus_save_pct_5v5, 4) AS pdo_5v5,
    ingested_at
FROM ranked
WHERE rn = 1
