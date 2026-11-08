WITH source_stats AS (
    SELECT * FROM {{ ref('bronze_team_stats') }}
),

ranked AS (
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
        ingested_at,
        ROW_NUMBER() OVER (PARTITION BY season, team_id ORDER BY ingested_at DESC) AS rn
    FROM source_stats
)

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
    ROUND(point_pct, 4) AS point_pct,
    ROUND(goals_for_per_game, 3) AS goals_for_per_game,
    ROUND(goals_against_per_game, 3) AS goals_against_per_game,
    ROUND(power_play_pct, 4) AS power_play_pct,
    ROUND(penalty_kill_pct, 4) AS penalty_kill_pct,
    ROUND(shots_for_per_game, 2) AS shots_for_per_game,
    ROUND(shots_against_per_game, 2) AS shots_against_per_game,
    ROUND(faceoff_win_pct, 4) AS faceoff_win_pct,
    ingested_at
FROM ranked
WHERE rn = 1
