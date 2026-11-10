import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import polars as pl
    import numpy as np

    return duckdb, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import
    """)
    return


@app.cell
def _(duckdb):
    con = duckdb.connect("../data/warehouse/dev.duckdb")
    df = con.sql("SELECT * FROM gold.gold_games").pl()
    print(df.head())
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory data analysis

    Lets start by checking data quality and then overal statistics
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Inspect rows that has missing values
    """)
    return


@app.cell
def _(df, pl):
    null_rows = df.filter(pl.any_horizontal(pl.all().is_null()))
    null_rows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are missing values, because the matches are preseason matches and those are first matches of the season. game_type is 1.

    Also team is totally different in preseason than start of the season, so we can drop preseason completely.

    https://en.wikipedia.org/wiki/Season_structure_of_the_NHL
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dataframe schema
    """)
    return


@app.cell
def _(df, pl):

    schema_overview = pl.DataFrame({
        "column_name": df.columns,
        "dtype": [str(t) for t in df.dtypes],
    })
    schema_overview
    return


@app.cell
def _(df, pl):
    # 1. Game Granularity & Chronology columns
    game_id_col = next((c for c in ["game_id", "id"] if c in df.columns), None)
    date_season_cols = [
        c for c in df.columns 
        if any(k in c.lower() for k in ["date", "season", "time", "timestamp"])
    ]
    team_perspective_cols = [
        c for c in df.columns 
        if any(k in c.lower() for k in ["home", "away", "team"])
    ]

    granularity_check = pl.DataFrame({
        "metric": [
            "Total rows",
            "Unique game IDs",
            "Granularity (One row per game?)",
            "Date / Season columns",
            "Home / Away columns",
        ],
        "value": [
            str(len(df)),
            str(df[game_id_col].n_unique()) if game_id_col else "N/A",
            str(len(df) == df[game_id_col].n_unique()) if game_id_col else "N/A",
            ", ".join(date_season_cols),
            ", ".join(team_perspective_cols),
        ],
    })

    granularity_check
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Filter Preseason Games & Define Season Coverage

    NHL games use `game_type` where:
    - `1` = Preseason
    - `2` = Regular season
    - `3` = Playoffs

    Preseason games feature non-standard rosters, experimental line-ups, and incomplete stats (causing null values). We filter out preseason matches to focus on competitive regular season and playoff games.
    """)
    return


@app.cell
def _(df, pl):
    # Check game_type distribution before filtering
    game_type_counts = (
        df.group_by("game_type")
        .agg([
            pl.len().alias("count"),
            pl.col("game_id").n_unique().alias("unique_games"),
        ])
        .sort("game_type")
    )
    game_type_counts
    return


@app.cell
def _(df, mo, pl):
    # Filter out preseason games
    filtered_games = df.filter(pl.col("game_type") != 1)

    # Check remaining nulls across all columns in filtered dataset
    remaining_nulls = (
        filtered_games
        .select([pl.col(c).is_null().sum().alias(c) for c in filtered_games.columns])
        .transpose(include_header=True, header_name="column_name", column_names=["null_count"])
        .filter(pl.col("null_count") > 0)
    )

    mo.md(f"""
    - **Total games before filtering:** {len(df):,}
    - **Total games after dropping preseason:** {len(filtered_games):,}
    - **Columns with nulls after filter:** {len(remaining_nulls)}
    """)
    return (filtered_games,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Structural Analysis

    Let's verify:
    1. **Granularity & Perspective:** Whether rows are game-centric (1 row per game with home & away) or team-centric (2 rows per game).
    2. **Chronology & Seasons:** Date columns, season coverage, and ordering.
    3. **Scores & Outcome Resolution:** Regulation vs Overtime (OT) vs Shootout (SO) recording.
    """)
    return


@app.cell
def _(filtered_games, pl):
    # 1. Perspective and Granularity
    total_filtered_rows = len(filtered_games)
    unique_filtered_games = filtered_games["game_id"].n_unique()
    is_game_centric = total_filtered_rows == unique_filtered_games

    # 2. Score, Outcome, and Timestamp column detection
    detected_score_cols = [
        c for c in filtered_games.columns 
        if any(k in c.lower() for k in ["score", "goal", "outcome", "winner", "win", "period", "ot", "so", "overtime", "shootout"])
    ]
    detected_date_cols = [
        c for c in filtered_games.columns 
        if any(k in c.lower() for k in ["date", "time", "season"])
    ]

    perspective_summary = pl.DataFrame({
        "Property": [
            "Total Filtered Rows",
            "Unique Game IDs",
            "Perspective",
            "Date & Chronology Columns",
            "Score & Outcome Columns",
        ],
        "Value": [
            str(total_filtered_rows),
            str(unique_filtered_games),
            "Single game per row (Home/Away columns)" if is_game_centric else "Team-centric (Multiple rows per game)",
            ", ".join(detected_date_cols),
            ", ".join(detected_score_cols),
        ],
    })
    perspective_summary
    return (detected_score_cols,)


@app.cell
def _(filtered_games, pl):
    # 3. Season coverage breakdown and chronological range
    season_coverage = (
        filtered_games
        .group_by("season", "game_type")
        .agg([
            pl.len().alias("games"),
            pl.col("game_date").min().alias("start_date") if "game_date" in filtered_games.columns else pl.lit("N/A").alias("start_date"),
            pl.col("game_date").max().alias("end_date") if "game_date" in filtered_games.columns else pl.lit("N/A").alias("end_date"),
        ])
        .sort(["season", "game_type"])
    )
    season_coverage
    return


@app.cell
def _(detected_score_cols, filtered_games):
    # 4. Score and Outcome sample preview
    score_sample_cols = list(
        dict.fromkeys(
            c
            for c in detected_score_cols + ["season", "game_date", "home_team_id", "away_team_id"]
            if c in filtered_games.columns
        )
    )

    filtered_games.select(score_sample_cols).head(10)
    return


if __name__ == "__main__":
    app.run()
