-- For every game whose team had metrics in the prior season under the same
-- lineage_id, the prior-season baseline must be populated. This catches
-- franchise-continuity regressions such as ARI (53) -> UTA (59).
WITH team_game_side AS (
    SELECT
        season,
        home_lineage_id AS lineage_id,
        home_prior_point_pct AS prior_point_pct
    FROM {{ ref('silver_games_enriched') }}

    UNION ALL

    SELECT
        season,
        away_lineage_id AS lineage_id,
        away_prior_point_pct AS prior_point_pct
    FROM {{ ref('silver_games_enriched') }}
),

expected AS (
    SELECT DISTINCT
        g.season,
        g.lineage_id
    FROM team_game_side g
    INNER JOIN {{ ref('silver_team_metrics') }} m
        ON m.season = g.season - 10001
        AND m.lineage_id = g.lineage_id
)

SELECT
    g.season,
    g.lineage_id,
    COUNT(*) AS games_with_null_prior
FROM team_game_side g
INNER JOIN expected e
    ON g.season = e.season
    AND g.lineage_id = e.lineage_id
WHERE g.prior_point_pct IS NULL
GROUP BY g.season, g.lineage_id
