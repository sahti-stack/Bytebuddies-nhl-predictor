-- =============================================================================
-- TEAM LINEAGE (FRANCHISE / HOCKEY-OPERATIONS CONTINUITY) CROSSWALK
-- =============================================================================
-- One row per official NHL team_id. Official identifiers are never overwritten;
-- this model only adds a synthetic lineage_id used for prior-season joins.
--
-- 2022-23 .. 2023-24 : Arizona Coyotes   (team_id 53, ARI)
-- 2024-25           : Utah Hockey Club  (team_id 59, UTA)
-- 2025-26+          : Utah Mammoth      (team_id 59, UTA)
--
-- Utah inherited Arizona's roster, coaching staff, prospects and hockey ops,
-- so prior-season features must follow the franchise lineage. The NHL kept
-- team_id = 59 across the Utah Hockey Club -> Utah Mammoth rebrand, therefore
-- joining on team_id (not team_name/team_abbrev) handles that rename.
-- =============================================================================
WITH latest_team AS (
    -- Latest known official identity for each team_id (handles renames).
    SELECT
        team_id,
        team_abbrev,
        team_name
    FROM (
        SELECT
            team_id,
            team_abbrev,
            team_name,
            ROW_NUMBER() OVER (
                PARTITION BY team_id
                ORDER BY season DESC
            ) AS rn
        FROM {{ ref('stg_team_stats') }}
    ) ranked
    WHERE rn = 1
),

lineage_overrides AS (
    -- (team_id, lineage_id, lineage_name)
    -- Add a row here for any future relocation/rebrand that keeps hockey ops.
    SELECT * FROM (VALUES
        (53, 53, 'Arizona Coyotes -> Utah Hockey Club -> Utah Mammoth'),
        (59, 53, 'Arizona Coyotes -> Utah Hockey Club -> Utah Mammoth')
    ) AS t(team_id, lineage_id, lineage_name)
)

SELECT
    t.team_id,
    t.team_abbrev AS latest_team_abbrev,
    t.team_name   AS latest_team_name,
    COALESCE(o.lineage_id, t.team_id) AS lineage_id,
    COALESCE(o.lineage_name, t.team_name) AS lineage_name,
    (o.team_id IS NOT NULL) AS is_lineage_override
FROM latest_team t
LEFT JOIN lineage_overrides o
    ON t.team_id = o.team_id
