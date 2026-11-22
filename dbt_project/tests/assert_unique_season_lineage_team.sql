-- A lineage_id must map to at most one official team per season.
-- Otherwise a prior-season join on (season, lineage_id) would duplicate games.
SELECT
    season,
    lineage_id,
    COUNT(DISTINCT team_id) AS team_count
FROM {{ ref('silver_team_metrics') }}
GROUP BY season, lineage_id
HAVING COUNT(DISTINCT team_id) > 1
