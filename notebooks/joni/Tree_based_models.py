import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import duckdb
    import marimo as mo
    import polars as pl
    import altair

    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    from sklearn.metrics import log_loss, roc_curve

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
    df = con.sql("""
        SELECT * 
        FROM gold.gold_games 
        WHERE season != 20212022
    """).pl()

    print(df.head())
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory data analysis

    Lets start by checking data quality and then overal statistics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dataframe schema
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):

    schema_overview = pl.DataFrame({
        "column_name": df.columns,
        "dtype": [str(t) for t in df.dtypes],
    })
    schema_overview
    return


@app.cell
def _(df, pl):
    # Get a list of only the columns that have at least one null
    null_cols = [col for col in df.columns if df[col].null_count() > 0]

    # Build a list 
    cols_to_show = list(
        dict.fromkeys(["season", "home_team_abbrev", "away_team_abbrev", "game_type"] + null_cols)
    )

    # filter rows that have nulls
    null_rows = df.filter(pl.any_horizontal(pl.col(null_cols).is_null())).select(
        cols_to_show
    )

    null_rows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have couple of null values that we have to solve:

    - We have to get 2021-2022 season stats to fill the prior columns of season 2022-2023. 2021-season will be only used to fill the prior numbers, not for training.
    - UTA team_abbrev has taken over ARI when they bought the team. Roster and management stayed the same, so we can keep them as same team in the dataset.
    """)
    return


@app.cell
def _(df):
    # Check if is_played is always true
    df["is_played"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature collection for tree based models

    Lets define this as binary problem:

    Target (y) = *home_win*

    ### Used only for splitting the data to train/validation/test sets, not a feature

    *season*

    *game_date*

    ### used as team identifier, not a feature:

    *home_team_abbrev*

    *away_team_abbrev*

    ## dropped features

    We drop playoffs and pre-season, only regular season is used

    *game_type = 2*

    Data has target leakage, model can use that to tell which team will win. We will drop them.

    *home_score* *away_score* *total_goals* *goal_diff* *home_win* *away_win* *is_regulation* *is_overtime*
    """)
    return


if __name__ == "__main__":
    app.run()
