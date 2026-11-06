"""Bronze layer ingestion from official NHL APIs into DuckDB."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Base URLs for official NHL endpoints
BASE_WEB_API = "https://api-web.nhle.com/v1"
BASE_STATS_API = "https://api.nhle.com/stats/rest/en"

# Fallback team list (used if dynamic team discovery fails)
ALL_TEAMS_FALLBACK = [
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

# Mapping of team full names to 3-letter abbreviations
TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}

# Explicit Polars schemas prevent runtime type inference issues
RAW_GAMES_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Int64,
    "season": pl.Int64,
    "game_type": pl.Int32,
    "game_date": pl.String,
    "start_time_utc": pl.String,
    "venue": pl.String,
    "game_state": pl.String,
    "home_team_id": pl.Int64,
    "home_team_abbrev": pl.String,
    "home_team_name": pl.String,
    "home_score": pl.Int32,
    "away_team_id": pl.Int64,
    "away_team_abbrev": pl.String,
    "away_team_name": pl.String,
    "away_score": pl.Int32,
    "outcome_last_period_type": pl.String,
    "raw_json": pl.String,
}

RAW_TEAM_STATS_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64,
    "team_id": pl.Int64,
    "team_abbrev": pl.String,
    "team_name": pl.String,
    "games_played": pl.Int32,
    "wins": pl.Int32,
    "losses": pl.Int32,
    "ot_losses": pl.Int32,
    "points": pl.Int32,
    "point_pct": pl.Float64,
    "goals_for_per_game": pl.Float64,
    "goals_against_per_game": pl.Float64,
    "power_play_pct": pl.Float64,
    "penalty_kill_pct": pl.Float64,
    "shots_for_per_game": pl.Float64,
    "shots_against_per_game": pl.Float64,
    "faceoff_win_pct": pl.Float64,
    "raw_json": pl.String,
}

RAW_TEAM_PERCENTAGES_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64,
    "team_id": pl.Int64,
    "team_abbrev": pl.String,
    "sat_pct": pl.Float64,
    "usat_pct": pl.Float64,
    "shooting_plus_save_pct_5v5": pl.Float64,
    "raw_json": pl.String,
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bronze_ingestion")


def create_http_session() -> requests.Session:
    """Create an HTTP session with retries and a descriptive User-Agent."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Bytebuddies-NHL-Predictor/1.0 (Student ML Project; KAMK)",
        "Accept": "application/json",
    })

    retries = Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def init_bronze_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the bronze schema and target tables if they do not exist."""
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")

    con.execute("""
        CREATE TABLE IF NOT EXISTS bronze.raw_games (
            game_id BIGINT PRIMARY KEY,
            season BIGINT,
            game_type INTEGER,
            game_date VARCHAR,
            start_time_utc VARCHAR,
            venue VARCHAR,
            game_state VARCHAR,
            home_team_id BIGINT,
            home_team_abbrev VARCHAR,
            home_team_name VARCHAR,
            home_score INTEGER,
            away_team_id BIGINT,
            away_team_abbrev VARCHAR,
            away_team_name VARCHAR,
            away_score INTEGER,
            outcome_last_period_type VARCHAR,
            raw_json VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS bronze.raw_team_stats (
            season BIGINT,
            team_id BIGINT,
            team_abbrev VARCHAR,
            team_name VARCHAR,
            games_played INTEGER,
            wins INTEGER,
            losses INTEGER,
            ot_losses INTEGER,
            points INTEGER,
            point_pct DOUBLE,
            goals_for_per_game DOUBLE,
            goals_against_per_game DOUBLE,
            power_play_pct DOUBLE,
            penalty_kill_pct DOUBLE,
            shots_for_per_game DOUBLE,
            shots_against_per_game DOUBLE,
            faceoff_win_pct DOUBLE,
            raw_json VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (season, team_id)
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS bronze.raw_team_percentages (
            season BIGINT,
            team_id BIGINT,
            team_abbrev VARCHAR,
            sat_pct DOUBLE,
            usat_pct DOUBLE,
            shooting_plus_save_pct_5v5 DOUBLE,
            raw_json VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (season, team_id)
        );
    """)
    logger.info("DuckDB bronze schema and tables verified.")


def insert_polars_into_duckdb(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    df: pl.DataFrame,
) -> None:
    """Insert or replace a Polars DataFrame into DuckDB, supporting Arrow or row fallback."""
    if df.is_empty():
        return

    # Try fast Arrow zero-copy registration first
    try:
        view_name = f"_tmp_{table_name.replace('.', '_')}"
        con.register(view_name, df)
        con.execute(f"""
            INSERT OR REPLACE INTO {table_name} BY NAME
            SELECT *, CURRENT_TIMESTAMP AS ingested_at
            FROM {view_name};
        """)
        con.unregister(view_name)
        return
    except Exception as e:
        logger.debug("Arrow registration unavailable (%s), using direct row insert.", e)

    # Robust fallback: Insert directly from Polars rows (works even without pyarrow)
    cols = df.columns
    col_str = ", ".join(cols) + ", ingested_at"
    placeholders = ", ".join(["?"] * len(cols)) + ", CURRENT_TIMESTAMP"
    sql = f"INSERT OR REPLACE INTO {table_name} ({col_str}) VALUES ({placeholders})"
    con.executemany(sql, df.rows())


def extract_team_name(team_dict: dict[str, Any] | None) -> str:
    """Safely extract team name from localized structures."""
    if not team_dict:
        return ""
    common = team_dict.get("commonName")
    if isinstance(common, dict):
        return common.get("default", "")
    if isinstance(common, str):
        return common
    name = team_dict.get("name")
    if isinstance(name, dict):
        return name.get("default", "")
    if isinstance(name, str):
        return name
    return str(team_dict.get("abbrev", ""))


def extract_team_abbrev_from_stat(item: dict[str, Any]) -> str:
    """Extract 3-letter team abbreviation from stats record."""
    abbr = item.get("teamAbbrevs") or item.get("teamAbbrev") or item.get("triCode")
    if abbr:
        return str(abbr)
    full_name = item.get("teamFullName")
    if full_name and full_name in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[full_name]
    return ""


def fetch_team_stats_for_season(
    session: requests.Session,
    season: str,
    delay: float = 0.2,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch seasonal team summary stats and return list of team records and team abbreviations."""
    url = f"{BASE_STATS_API}/team/summary?cayenneExp=seasonId={season}%20and%20gameTypeId=2"
    logger.info("Fetching team stats for season %s...", season)
    try:
        resp = session.get(url, timeout=15)
        time.sleep(delay)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            team_abbrevs: list[str] = []
            for item in data:
                abbr = extract_team_abbrev_from_stat(item)
                if abbr and abbr not in team_abbrevs:
                    team_abbrevs.append(abbr)
            logger.info("Found %d teams for season %s.", len(team_abbrevs), season)
            return data, (team_abbrevs if team_abbrevs else ALL_TEAMS_FALLBACK)

        logger.warning(
            "Failed to fetch team stats (status %d): %s",
            resp.status_code,
            resp.text[:100],
        )
    except requests.RequestException as e:
        logger.error("Network error fetching team stats for season %s: %s", season, e)

    return [], ALL_TEAMS_FALLBACK


def fetch_team_percentages_for_season(
    session: requests.Session,
    season: str,
    delay: float = 0.2,
) -> list[dict[str, Any]]:
    """Fetch advanced percentages (Corsi/satPct, Fenwick/usatPct, PDO)."""
    url = f"{BASE_STATS_API}/team/percentages?cayenneExp=seasonId={season}%20and%20gameTypeId=2"
    logger.info("Fetching team percentages (Corsi, Fenwick, PDO) for season %s...", season)
    try:
        resp = session.get(url, timeout=15)
        time.sleep(delay)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            logger.info("Found %d percentage records for season %s.", len(data), season)
            return data
    except requests.RequestException as e:
        logger.error("Network error fetching team percentages for season %s: %s", season, e)
    return []


def fetch_games_for_team_season(
    session: requests.Session,
    team: str,
    season: str,
    delay: float = 0.15,
) -> list[dict[str, Any]]:
    """Fetch full season schedule for a given team."""
    url = f"{BASE_WEB_API}/club-schedule-season/{team}/{season}"
    try:
        resp = session.get(url, timeout=15)
        time.sleep(delay)
        if resp.status_code == 200:
            return resp.json().get("games", [])
        if resp.status_code == 404:
            logger.debug("Schedule not found for %s (%s)", team, season)
        else:
            logger.warning(
                "Error fetching schedule for %s (%s): status %d",
                team,
                season,
                resp.status_code,
            )
    except requests.RequestException as e:
        logger.error("Network error fetching schedule for %s (%s): %s", team, season, e)
    return []


def ingest_season(
    session: requests.Session,
    con: duckdb.DuckDBPyConnection,
    season: str,
    delay: float = 0.15,
) -> tuple[int, int, int]:
    """Ingest games, team stats, and advanced percentages for a single season."""
    logger.info("=== Starting ingestion for Season %s ===", season)

    # 1. Fetch team stats and save with explicit schema
    stats_data, teams_to_fetch = fetch_team_stats_for_season(session, season, delay=delay)
    if stats_data:
        parsed_stats = [
            {
                "season": int(s.get("seasonId", season)),
                "team_id": s.get("teamId"),
                "team_abbrev": extract_team_abbrev_from_stat(s),
                "team_name": s.get("teamFullName"),
                "games_played": s.get("gamesPlayed"),
                "wins": s.get("wins"),
                "losses": s.get("losses"),
                "ot_losses": s.get("otLosses"),
                "points": s.get("points"),
                "point_pct": s.get("pointPct"),
                "goals_for_per_game": s.get("goalsForPerGame"),
                "goals_against_per_game": s.get("goalsAgainstPerGame"),
                "power_play_pct": s.get("powerPlayPct"),
                "penalty_kill_pct": s.get("penaltyKillPct"),
                "shots_for_per_game": s.get("shotsForPerGame"),
                "shots_against_per_game": s.get("shotsAgainstPerGame"),
                "faceoff_win_pct": s.get("faceoffWinPct"),
                "raw_json": json.dumps(s),
            }
            for s in stats_data
        ]
        df_stats = pl.DataFrame(parsed_stats, schema=RAW_TEAM_STATS_SCHEMA)
        insert_polars_into_duckdb(con, "bronze.raw_team_stats", df_stats)
        logger.info("Saved %d records to bronze.raw_team_stats.", len(parsed_stats))

    # 2. Fetch team percentages and save with explicit schema
    pct_data = fetch_team_percentages_for_season(session, season, delay=delay)
    if pct_data:
        parsed_pct = [
            {
                "season": int(p.get("seasonId", season)),
                "team_id": p.get("teamId"),
                "team_abbrev": extract_team_abbrev_from_stat(p),
                "sat_pct": p.get("satPct"),
                "usat_pct": p.get("usatPct"),
                "shooting_plus_save_pct_5v5": p.get("shootingPlusSavePct5v5"),
                "raw_json": json.dumps(p),
            }
            for p in pct_data
        ]
        df_pct = pl.DataFrame(parsed_pct, schema=RAW_TEAM_PERCENTAGES_SCHEMA)
        insert_polars_into_duckdb(con, "bronze.raw_team_percentages", df_pct)
        logger.info("Saved %d records to bronze.raw_team_percentages.", len(parsed_pct))

    # 3. Fetch schedules for each team and deduplicate games by game_id
    games_by_id: dict[int, dict[str, Any]] = {}
    total_teams = len(teams_to_fetch)

    for idx, team in enumerate(teams_to_fetch, 1):
        logger.info(
            "[%d/%d] Fetching schedule for team %s (%s)...",
            idx,
            total_teams,
            team,
            season,
        )
        raw_games = fetch_games_for_team_season(session, team, season, delay=delay)

        for g in raw_games:
            gid = g.get("id")
            if not gid or gid in games_by_id:
                continue

            venue_obj = g.get("venue")
            venue_str = (
                venue_obj.get("default")
                if isinstance(venue_obj, dict)
                else str(venue_obj or "")
            )

            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            outcome = g.get("gameOutcome", {})

            games_by_id[gid] = {
                "game_id": gid,
                "season": int(g.get("season", season)),
                "game_type": g.get("gameType"),
                "game_date": g.get("gameDate"),
                "start_time_utc": g.get("startTimeUTC"),
                "venue": venue_str,
                "game_state": g.get("gameState"),
                "home_team_id": home.get("id"),
                "home_team_abbrev": home.get("abbrev"),
                "home_team_name": extract_team_name(home),
                "home_score": home.get("score"),
                "away_team_id": away.get("id"),
                "away_team_abbrev": away.get("abbrev"),
                "away_team_name": extract_team_name(away),
                "away_score": away.get("score"),
                "outcome_last_period_type": (
                    outcome.get("lastPeriodType")
                    if isinstance(outcome, dict)
                    else None
                ),
                "raw_json": json.dumps(g),
            }

    unique_games = list(games_by_id.values())
    if unique_games:
        df_games = pl.DataFrame(unique_games, schema=RAW_GAMES_SCHEMA)
        insert_polars_into_duckdb(con, "bronze.raw_games", df_games)
        logger.info(
            "Saved %d unique games for Season %s to bronze.raw_games.",
            len(unique_games),
            season,
        )

    return len(unique_games), len(stats_data), len(pct_data)


def run_ingestion(
    db_path: str = "data/warehouse/dev.duckdb",
    seasons: list[str] | None = None,
    delay: float = 0.15,
) -> None:
    """Run full bronze ingestion pipeline with context management."""
    if seasons is None:
        # 3 previous completed seasons + current ongoing 2026-2027 season
        seasons = ["20232024", "20242025", "20252026", "20262027"]

    resolved_path = Path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Connecting to DuckDB at: %s", resolved_path.resolve())

    start_time = time.time()

    # Use context managers for deterministic cleanup
    with duckdb.connect(str(resolved_path)) as con:
        init_bronze_schema(con)

        with create_http_session() as session:
            for season in seasons:
                ingest_season(session, con, season, delay=delay)

        # Verification query
        cnt_games = con.execute("SELECT count(*) FROM bronze.raw_games;").fetchone()[0]
        cnt_stats = con.execute("SELECT count(*) FROM bronze.raw_team_stats;").fetchone()[0]
        cnt_pcts = con.execute("SELECT count(*) FROM bronze.raw_team_percentages;").fetchone()[0]

    elapsed = time.time() - start_time
    logger.info("==================================================")
    logger.info("BRONZE INGESTION COMPLETE in %.1f seconds!", elapsed)
    logger.info("Total rows in bronze.raw_games:            %d", cnt_games)
    logger.info("Total rows in bronze.raw_team_stats:       %d", cnt_stats)
    logger.info("Total rows in bronze.raw_team_percentages: %d", cnt_pcts)
    logger.info("==================================================")


def main() -> None:
    """CLI entry point for bronze ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest NHL raw data into DuckDB Bronze layer",
    )
    parser.add_argument(
        "--db-path",
        default="data/warehouse/dev.duckdb",
        help="Path to DuckDB database file (default: data/warehouse/dev.duckdb)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["20232024", "20242025", "20252026", "20262027"],
        help="Seasons to ingest (e.g. 20242025 20252026 20262027)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Delay in seconds between HTTP requests (default: 0.15)",
    )
    args = parser.parse_args()

    run_ingestion(db_path=args.db_path, seasons=args.seasons, delay=args.delay)


if __name__ == "__main__":
    main()
