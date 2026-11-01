import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests
    import polars as pl
    import time

    return mo, pl, requests, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏒 NHL API Exploration Notebook
    **Author:** Panu Eklund
    **SCRUM tasks covered:** SCRUM-14, SCRUM-15, SCRUM-16, SCRUM-17

    ---

    ## 📖 SCRUM-14 – API Documentation Summary

    The NHL exposes two base URLs:

    | Base URL | Purpose |
    |---|---|
    | `https://api-web.nhle.com/v1/` | Schedule, scores, rosters, game centre |
    | `https://api.nhle.com/stats/rest/en/` | Aggregated stats (skaters, goalies, teams) |

    ### Key conventions
    - **`TEAM_ABBR`** – 3-letter team code, e.g. `TOR`, `WSH`, `BOS`
    - **`SEASON_ID`** – 8-digit format `<startYear><endYear>`, e.g. `20232024`
    - **`GAME_ID`** – season year prefix + unique match id, e.g. `2023020001`

    ### Main endpoint groups

    | Group | Example endpoint |
    |---|---|
    | Schedule | `GET /v1/schedule/now` |
    | Club schedule | `GET /v1/club-schedule-season/TEAM_ABBR/now` |
    | Standings | `GET /v1/standings/now` |
    | Roster | `GET /v1/roster/TEAM_ABBR/SEASON_ID` |
    | Game boxscore | `GET /v1/gamecenter/GAME_ID/boxscore` |
    | Play-by-play | `GET /v1/gamecenter/GAME_ID/play-by-play` |
    | Skater stats | `GET /stats/rest/en/skater/summary?...` |
    | Goalie stats | `GET /stats/rest/en/goalie/summary?...` |
    | Team stats | `GET /stats/rest/en/team/summary?...` |
    | Player search | `GET https://search.d3.nhle.com/api/v1/search/player?q=STRING` |

    Full documentation: https://gitlab.com/dword4/nhlapi/-/blob/master/new-api.md
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧪 SCRUM-15 – Simple Test Requests

    Let's verify the API is reachable and inspect a few core endpoints.
    """)
    return


@app.cell
def _(requests):
    BASE_WEB = "https://api-web.nhle.com/v1"
    BASE_STATS = "https://api.nhle.com/stats/rest/en"

    def get_json(url: str) -> dict | None:
        """Fetch JSON from url, return dict or None on failure."""
        try:
            r = requests.get(url, timeout=10)
            print(f"  [{r.status_code}] {url}")
            if r.status_code == 200:
                return r.json()
            else:
                print(f"  ERROR: {r.text[:200]}")
                return None
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            return None

    return BASE_STATS, BASE_WEB, get_json


@app.cell
def _(mo):
    mo.md("""
    ### Test 1 – Today's schedule (`/schedule/now`)
    """)
    return


@app.cell
def _(BASE_WEB, get_json, pl):
    schedule_now = get_json(f"{BASE_WEB}/schedule/now")

    if schedule_now:
        games_today = []
        for week in schedule_now.get("gameWeek", []):
            for game in week.get("games", []):
                games_today.append({
                    "date": week["date"],
                    "game_id": game.get("id"),
                    "home": game.get("homeTeam", {}).get("abbrev"),
                    "away": game.get("awayTeam", {}).get("abbrev"),
                    "state": game.get("gameState"),
                    "venue": game.get("venue"),
                })
        df_schedule_now = pl.DataFrame(games_today) if games_today else pl.DataFrame()
        print(f"Games found in current week: {len(games_today)}")
    else:
        df_schedule_now = pl.DataFrame()

    df_schedule_now
    return


@app.cell
def _(mo):
    mo.md("""
    ### Test 2 – Current Standings (`/standings/now`)
    """)
    return


@app.cell
def _(BASE_WEB, get_json, pl):
    standings_data = get_json(f"{BASE_WEB}/standings/now")

    if standings_data:
        records = standings_data.get("standings", [])
        df_standings = pl.DataFrame([
            {
                "team": r.get("teamAbbrev", {}).get("default"),
                "full_name": r.get("teamName", {}).get("default"),
                "division": r.get("divisionName"),
                "conference": r.get("conferenceName"),
                "wins": r.get("wins"),
                "losses": r.get("losses"),
                "otLosses": r.get("otLosses"),
                "points": r.get("points"),
                "games_played": r.get("gamesPlayed"),
                "goals_for": r.get("goalFor"),
                "goals_against": r.get("goalAgainst"),
                "goal_diff": r.get("goalDifferential"),
                "home_wins": r.get("homeWins"),
                "road_wins": r.get("roadWins"),
                "streak_type": r.get("streakCode"),
                "streak_count": r.get("streakCount"),
            }
            for r in records
        ])
        print(f"Teams in standings: {len(df_standings)}")
        print(f"Available columns: {df_standings.columns}")
    else:
        df_standings = pl.DataFrame()

    df_standings
    return


@app.cell
def _(mo):
    mo.md("""
    ### Test 3 – Team Roster (`/roster/TOR/20242025`)
    """)
    return


@app.cell
def _(BASE_WEB, get_json, pl):
    roster_data = get_json(f"{BASE_WEB}/roster/TOR/20242025")

    if roster_data:
        skaters = []
        for pos_group in ["forwards", "defensemen", "goalies"]:
            for p in roster_data.get(pos_group, []):
                skaters.append({
                    "player_id": p.get("id"),
                    "first_name": p.get("firstName", {}).get("default"),
                    "last_name": p.get("lastName", {}).get("default"),
                    "position": p.get("positionCode"),
                    "jersey": p.get("sweaterNumber"),
                    "shoots_catches": p.get("shootsCatches"),
                    "height_cm": p.get("heightInCentimeters"),
                    "weight_kg": p.get("weightInKilograms"),
                    "birth_date": p.get("birthDate"),
                    "birth_country": p.get("birthCountry"),
                })
        df_roster = pl.DataFrame(skaters)
        print(f"Players on roster: {len(df_roster)}")
    else:
        df_roster = pl.DataFrame()

    df_roster
    return


@app.cell
def _(mo):
    mo.md("""
    ### Test 4 – Game Boxscore (`/gamecenter/{GAME_ID}/boxscore`)
    """)
    return


@app.cell
def _(BASE_WEB, get_json):
    # A known completed game (2023-24 regular season opener)
    GAME_ID = "2023020001"
    boxscore_data = get_json(f"{BASE_WEB}/gamecenter/{GAME_ID}/boxscore")

    if boxscore_data:
        home = boxscore_data.get("homeTeam", {})
        away = boxscore_data.get("awayTeam", {})
        print(f"\nGame: {away.get('abbrev')} @ {home.get('abbrev')}")
        print(f"Score: {away.get('score')} - {home.get('score')}")
        print(f"Top-level keys: {list(boxscore_data.keys())}")
        player_by_game_stats = boxscore_data.get("playerByGameStats", {})
        print(f"\nPlayer stats keys: {list(player_by_game_stats.keys())}")
    else:
        print("Could not fetch boxscore")

    boxscore_data
    return (GAME_ID,)


@app.cell
def _(mo):
    mo.md("""
    ### Test 5 – Team Summary Stats (`/stats/rest/en/team/summary`)
    """)
    return


@app.cell
def _(BASE_STATS, get_json, pl):
    team_stats_url = (
        f"{BASE_STATS}/team/summary"
        "?cayenneExp=seasonId=20232024%20and%20gameTypeId=2"
        "&sort=points&dir=DESC&limit=32"
    )
    team_stats_data = get_json(team_stats_url)

    if team_stats_data:
        df_team_stats = pl.DataFrame(team_stats_data.get("data", []))
        # Move team name columns to the front for readability
        name_cols = [c for c in ["teamFullName", "teamAbbrevs"] if c in df_team_stats.columns]
        other_cols = [c for c in df_team_stats.columns if c not in name_cols]
        df_team_stats = df_team_stats.select(name_cols + other_cols)
        print(f"Rows: {len(df_team_stats)}, Columns: {df_team_stats.columns}")
    else:
        df_team_stats = pl.DataFrame()

    df_team_stats
    return


@app.cell
def _(mo):
    mo.md("""
    ### Test 6 – Top Skater Stats (`/stats/rest/en/skater/summary`)
    """)
    return


@app.cell
def _(BASE_STATS, get_json, pl):
    skater_url = (
        f"{BASE_STATS}/skater/summary"
        "?cayenneExp=seasonId=20232024%20and%20gameTypeId=2"
        "&sort=points&dir=DESC&limit=20"
    )
    skater_data = get_json(skater_url)

    if skater_data:
        df_skaters = pl.DataFrame(skater_data.get("data", []))
        # Move player name columns to the front for readability
        name_cols = [c for c in ["skaterFullName", "teamAbbrevs"] if c in df_skaters.columns]
        other_cols = [c for c in df_skaters.columns if c not in name_cols]
        df_skaters = df_skaters.select(name_cols + other_cols)
        print(f"Rows: {len(df_skaters)}")
        print(f"Columns: {df_skaters.columns}")
    else:
        df_skaters = pl.DataFrame()

    df_skaters
    return


@app.cell
def _(mo):
    mo.md("""
    ### Test 7 – Play-by-Play (`/gamecenter/{GAME_ID}/play-by-play`)
    """)
    return


@app.cell
def _(BASE_WEB, GAME_ID, get_json, pl):
    pbp_data = get_json(f"{BASE_WEB}/gamecenter/{GAME_ID}/play-by-play")

    if pbp_data:
        plays = pbp_data.get("plays", [])
        print(f"Total play events: {len(plays)}")
        if plays:
            print(f"\nFirst play keys: {list(plays[0].keys())}")
            print(f"First play: {plays[0]}")

        df_pbp = pl.DataFrame([
            {
                "event_id": p.get("eventId"),
                "period": p.get("periodDescriptor", {}).get("number"),
                "time_in_period": p.get("timeInPeriod"),
                "type_code": p.get("typeCode"),
                "type_desc": p.get("typeDescKey"),
                "zone": p.get("details", {}).get("zoneCode") if p.get("details") else None,
                "x_coord": p.get("details", {}).get("xCoord") if p.get("details") else None,
                "y_coord": p.get("details", {}).get("yCoord") if p.get("details") else None,
                "shot_type": p.get("details", {}).get("shotType") if p.get("details") else None,
            }
            for p in plays
        ])
        print(f"\nPlay types found: {df_pbp['type_desc'].unique().to_list()}")
    else:
        df_pbp = pl.DataFrame()

    df_pbp.head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 📊 SCRUM-17 – Available Data Overview

    Based on the test requests above, here is a structured summary of what data is accessible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Schedule / Game data
    | Field | Source endpoint |
    |---|---|
    | Game ID, Season, Game type | `/schedule/now` |
    | Home team, Away team, Venue | `/schedule/now` |
    | Start time (UTC + venue timezone) | `/schedule/now` |
    | Game state (FINAL, LIVE, FUT…) | `/schedule/now` |
    | Outcome (REG / OT / SO) | `/schedule/now` + `/boxscore` |
    | Winning goalie, Winning goal scorer | `/schedule/now` |
    | TV broadcasts by market | `/schedule/now` |

    ### Standings data
    | Field | Source endpoint |
    |---|---|
    | Wins, Losses, OT Losses | `/standings/now` |
    | Points, Games Played | `/standings/now` |
    | Goals For / Against, Goal Diff | `/standings/now` |
    | Home record, Road record | `/standings/now` |
    | Current streak | `/standings/now` |
    | Division & Conference | `/standings/now` |

    ### Team aggregate stats (per season)
    | Field | Source endpoint |
    |---|---|
    | Goals For/Against per game | `/stats/rest/en/team/summary` |
    | Shots For/Against per game | `/stats/rest/en/team/summary` |
    | Power play %, Penalty kill % | `/stats/rest/en/team/powerplay` |
    | Faceoff win % | `/stats/rest/en/team/faceoffpercentages` |
    | Days between games | `/stats/rest/en/team/daysbetweengames` |
    | Goals by period | `/stats/rest/en/team/goalsbyperiod` |

    ### Roster data
    | Field | Source endpoint |
    |---|---|
    | Player ID, Name, Position | `/roster/TEAM_ABBR/SEASON_ID` |
    | Jersey number, Height, Weight | `/roster/TEAM_ABBR/SEASON_ID` |
    | Birth date, Birth country | `/roster/TEAM_ABBR/SEASON_ID` |
    | Shoots/Catches hand | `/roster/TEAM_ABBR/SEASON_ID` |

    ### Individual player/skater stats
    | Field | Source endpoint |
    |---|---|
    | Goals, Assists, Points | `/stats/rest/en/skater/summary` |
    | +/-, PIM, shots, shooting % | `/stats/rest/en/skater/summary` |
    | TOI, PP/SH/EV splits | `/stats/rest/en/skater/timeonice` |
    | Faceoff %, scoring rates | multiple report types |

    ### Play-by-play (event-level)
    | Field | Source endpoint |
    |---|---|
    | Event type (shot, goal, hit…) | `/gamecenter/GAME_ID/play-by-play` |
    | Period & time in period | `/gamecenter/GAME_ID/play-by-play` |
    | Rink coordinates (x, y) | `/gamecenter/GAME_ID/play-by-play` |
    | Zone code (O/N/D) | `/gamecenter/GAME_ID/play-by-play` |
    | Shot type (wrist, slap…) | `/gamecenter/GAME_ID/play-by-play` |
    | Player IDs involved | `/gamecenter/GAME_ID/play-by-play` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ⚠️ SCRUM-16 – Potential Limitations

    ### 1. Rate Limiting
    - The NHL API has **no published rate limit** in the documentation.
    - In practice, aggressive scraping (hundreds of requests/sec) may result in `429 Too Many Requests` or temporary IP bans.
    - **Recommendation:** Add a `time.sleep(0.5)` delay between requests when bulk-fetching historical seasons.

    ### 2. Authentication
    - The API is **publicly accessible** with no API key required.
    - This is unofficial/undocumented — the NHL could add auth or block access at any time.

    ### 3. Missing / Incomplete Fields
    - **Live game timing details** (exact clock when game ends) are gone in the new API.
    - **Odds / betting data** is returned from `/schedule/now` but depends on the user's country.
    - **`gameType` codes** are not always documented: `1=Preseason`, `2=Regular`, `3=Playoffs` (inferred).
    - Some historical games may have `null` coordinates in play-by-play.

    ### 4. Data Coverage
    - Older seasons (pre-2017) may have incomplete play-by-play data.
    - The stats API uses `cayenneExp` query parameter — a custom filter syntax that is not well documented.
    - **Pagination**: Stats endpoints return a `total` field; use `start` + `limit` to paginate large result sets.

    ### 5. Unofficial API / Stability
    - This is a **reverse-engineered API** — no SLA, no versioning guarantee.
    - The API changed significantly ~2023 (hence "new-api.md") and may change again.
    - **Recommendation:** Pin tested endpoint URLs and validate responses in the pipeline.

    ### 6. Missing features for ML prediction
    | Feature desired | Available? | Notes |
    |---|---|---|
    | Home/Away game context | ✅ Yes | Via schedule endpoints |
    | Recent form (last N games) | ✅ Yes | Calculate from club-schedule |
    | Head-to-head history | ⚠️ Partial | Requires filtering historical schedule |
    | Player injuries | ❌ No | Not in the public API |
    | Line combinations / ice time | ✅ Yes | Via TOI skater stats |
    | Goalie save % by strength | ✅ Yes | Via `/goalie/savesByStrength` |
    | Expected goals (xG) | ❌ No | Not provided; must compute from PBP |
    | Betting odds | ⚠️ Partial | Only for supported countries |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Rate limit probe – 10 rapid sequential requests
    Sends 10 rapid requests to measure response behaviour. All 200s = no immediate rate limiting.
    """)
    return


@app.cell
def _(BASE_WEB, requests, time):
    _results = []
    for _i in range(10):
        _start = time.perf_counter()
        _r = requests.get(f"{BASE_WEB}/standings/now", timeout=10)
        _elapsed = time.perf_counter() - _start
        _results.append({
            "request": _i + 1,
            "status": _r.status_code,
            "response_ms": round(_elapsed * 1000, 1),
        })

    import polars as _pl
    df_rate_probe = _pl.DataFrame(_results)
    print(df_rate_probe)
    df_rate_probe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Historical data availability by season (2015–2025)
    Checks which seasons are accessible by querying mid-season standings.
    """)
    return


@app.cell
def _(BASE_WEB, get_json, pl, time):
    _seasons = [
        ("20152016", "2016-01-15"),
        ("20162017", "2017-01-15"),
        ("20172018", "2018-01-15"),
        ("20182019", "2019-01-15"),
        ("20192020", "2020-01-15"),
        ("20202021", "2021-01-15"),
        ("20212022", "2022-01-15"),
        ("20222023", "2023-01-15"),
        ("20232024", "2024-01-15"),
        ("20242025", "2025-01-15"),
    ]

    _rows = []
    for _season_id, _check_date in _seasons:
        _data = get_json(f"{BASE_WEB}/standings/{_check_date}")
        time.sleep(0.3)
        if _data:
            _teams = len(_data.get("standings", []))
            _rows.append({"season": _season_id, "teams_in_standings": _teams, "available": True})
        else:
            _rows.append({"season": _season_id, "teams_in_standings": 0, "available": False})

    df_history = pl.DataFrame(_rows)
    print(df_history)
    df_history
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✅ Summary & Conclusions

    ### What we confirmed works
    - Schedule, standings, roster, boxscore, play-by-play, team stats, skater stats — **all accessible without auth**
    - Historical data is available back at least to the **2015-16 season**
    - The stats API supports rich filtering via `cayenneExp` and multiple report types (summary, powerplay, faceoff, TOI…)

    ### Key data fields for the ML predictor
    1. **Team form**: wins/losses over last N games from `/club-schedule-season`
    2. **Standings points**: direct from `/standings`
    3. **Team stats**: shots, goals, PP%, PK%, faceoff% from `/stats/rest/en/team/summary`
    4. **Goalie stats**: save % from `/stats/rest/en/goalie/summary`
    5. **Home/Away flag**: from schedule data
    6. **Play-by-play**: shot locations + types for xG proxy

    ### Recommended next steps
    - Build a data pipeline that collects all regular season games for seasons 2018–2025
    - Store raw JSON in DuckDB (via dbt) for reproducibility
    - Implement polite request pacing (~0.5s between calls)
    - Monitor for 429 / 5xx and implement exponential back-off
    """)
    return


if __name__ == "__main__":
    app.run()
