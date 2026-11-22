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
def _(df):
    # Check null counts
    nulls = df.null_count()
    nulls.select([col for col in nulls.columns if nulls[col][0] > 0])
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

    ### used as team identifier:

    *home_team_abbrev*
    *away_team_abbrev*
    """)
    return


if __name__ == "__main__":
    app.run()
