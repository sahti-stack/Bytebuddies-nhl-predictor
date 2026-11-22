WITH stats AS (
    SELECT * FROM {{ ref('stg_team_stats') }}
),

percentages AS (
    SELECT * FROM {{ ref('stg_team_percentages') }}
),

lineage AS (
    SELECT * FROM {{ ref('silver_team_lineage') }}
)

SELECT
    s.season,
    s.team_id,
    s.team_abbrev,
    s.team_name,
    COALESCE(l.lineage_id, s.team_id) AS lineage_id,
    s.games_played,
    s.wins,
    s.losses,
    s.ot_losses,
    s.points,
    s.point_pct,
    s.goals_for_per_game,
    s.goals_against_per_game,
    s.power_play_pct,
    s.penalty_kill_pct,
    s.shots_for_per_game,
    s.shots_against_per_game,
    s.faceoff_win_pct,
    p.sat_pct AS corsi_pct,
    p.usat_pct AS fenwick_pct,
    p.pdo_5v5,
    s.ingested_at
FROM stats s
LEFT JOIN percentages p
    ON s.season = p.season
    AND s.team_id = p.team_id
LEFT JOIN lineage l
    ON s.team_id = l.team_id
