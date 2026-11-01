import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests
    import polars as pl

    return mo, pl, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Testing how the API works and going trough where is needed information
    """)
    return


@app.cell
def _(pl, requests):
    #Get club stats by Season and Game Type
    response = requests.get("https://api-web.nhle.com/v1/club-stats/TOR/20232024/2")
    print(response.status_code)
    print(response.content)
    df = pl.read_json(response.content)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, pl):
    #exlude goalies, and explode nested struct column. Get names from one nested information pile.
    df_players = (
        df
        .select(pl.col("*").exclude("goalies"))
        .explode("skaters")
        .unnest("skaters")
        .with_columns([
            pl.col("firstName").struct.field("default").alias("firstName"),
            pl.col("lastName").struct.field("default").alias("lastName"),
        ])
    )

    df_players
    return


@app.cell
def _(pl, requests):
    #Get club schelude per season
    response_schedule = requests.get("https://api-web.nhle.com/v1/club-schedule-season/TOR/20232024")
    print(response_schedule.status_code)
    print(response_schedule.content)
    df_schedule = pl.read_json(response_schedule.content)
    return (df_schedule,)


@app.cell
def _(df_schedule):
    df_schedule
    return


@app.cell
def _(df_schedule, pl):
    #reach to commonName to get team name and id
    df_games = (
        df_schedule
        .explode("games")
        .unnest("games")
        .with_columns(
            home_score=pl.col("homeTeam").struct.field("score"),
            home_team=pl.col("homeTeam").struct.field("commonName").struct.field("default"),
            home_team_id=pl.col("homeTeam").struct.field("id"),
            away_score=pl.col("awayTeam").struct.field("score"),
            away_team=pl.col("awayTeam").struct.field("commonName").struct.field("default"),
            away_team_id=pl.col("awayTeam").struct.field("id")
        )
    )
    #df_games
    df_games.select(["id", "home_team", "home_team_id", "home_score", "away_team", "away_team_id", "away_score"])
    return


@app.cell
def _(pl, requests):
    #Get season schedule as now
    response_schedule_season = requests.get("https://api-web.nhle.com/v1/schedule-calendar/now")
    print(response_schedule_season.status_code)
    print(response_schedule_season.content)
    df_schedule_season = pl.read_json(response_schedule_season.content)
    return (df_schedule_season,)


@app.cell
def _(df_schedule_season):
    df_schedule_season.head(10)
    return


@app.cell
def _(df_schedule_season):
    #just testing what is inside of this
    df_now = (
        df_schedule_season
        .explode("teams")
        .unnest("teams")
    )

    df_now
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gettings teams playing per season
    """)
    return


@app.cell
def _(requests):
    seasons_to_check = {
        "20222023": "2023-01-15",
        "20232024": "2024-01-15",
        "20242025": "2025-01-15",
    }

    teams_per_season = {}

    for _season, _date in seasons_to_check.items():
      _url = f"https://api-web.nhle.com/v1/standings/{_date}"
      _response = requests.get(_url)

      if _response.status_code == 200:
        _data = _response.json()

        _season_teams = []
        for _record in _data.get("standings", []):
          _abbr = _record.get("teamAbbrev", {}).get("default")
          if _abbr:
            _season_teams.append(_abbr)

        teams_per_season[_season] = sorted(_season_teams)
        print(f"loytyi {len(_season_teams)} joukkuetta kaudelle {_season}")
      else:
        print(f"virhe {_season} haussa: Status {_response.status_code}")

    teams_per_season
    return (teams_per_season,)


@app.cell
def _(teams_per_season):
    print(teams_per_season.get("20232024"))
    return


if __name__ == "__main__":
    app.run()
