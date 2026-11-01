import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests
    import polars as pl

    return pl, requests


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


if __name__ == "__main__":
    app.run()
