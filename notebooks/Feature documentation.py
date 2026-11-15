import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Documentation of used and dropped features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pre-season dropped

    NHL games use `game_type` where:
    - `1` = Preseason
    - `2` = Regular season
    - `3` = Playoffs

    Preseason matches and line-ups differ from regular season games so much, so it will skew the data. Games are played at neutral sites and there is incent to get best points. Pre-season works for coaches tinker with line-up, test tryouts. Data has total of 4523 matches and 325 of them are preaseason matches, thats 7%. For these reasons, we drop the preseason matches from the dataset.

    https://en.wikipedia.org/wiki/Season_structure_of_the_NHL
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Home-win advantage as baseline

    Home team advantage is really important baseline factor when evaluating accuracy in National Hockey League matches. In NHL the average winning percentage is 54,5% (1) There are no systematic variation in home advantage across teams and additional rest for the home team has no impact on the the win-loss probability. (2)

    In our dataset that includes matches from 2022 to 2025. Homewin-rate varies from 52% to 56,15% between seasons. Because home team wins more often than away teams, naive "always-predict-home" strategy is more realistic baseline than random guessing 50%/50%.

    Predicting always average home-win rate of 54% gives a log loss of about 0.69, which is slightly better than a 0.693 coin-flip. (3) So if our model log loss is get lower than that, it has learned from the team and match data rather than just guessing home team.

    1. Swartz TB, Arce A. New insights involving the home team advantage. Saatavilla: https://www.sfu.ca/~tswartz/papers/hca.pdf
    2. Leard B, Doyle JM. Variations in home advantage: evidence from the National Hockey League. Saatavilla: https://www.degruyterbrill.com/document/doi/10.1515/1559-0410.1446/html
    3. Porela M. Predicting NHL match outcomes using a data-driven model [master's thesis]. Espoo: Aalto University; 2024. saatavilla: https://aaltodoc.aalto.fi/server/api/core/bitstreams/c3eb25e6-a001-467d-94d6-0b6fdab2f432/content
    """)
    return


if __name__ == "__main__":
    app.run()
