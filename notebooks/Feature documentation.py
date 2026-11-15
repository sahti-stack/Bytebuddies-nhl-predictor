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

    Preseason matches and line-ups differ from regular season games so much, so it will skew the data. Games are played at neutral sites and there is incent to get best points. Pre-season works for coaches tinker with line-up, test tryouts. Data has total of 4523 matches and 325 of them are preaseason matches, thats 7%. For these reasons, we drop the preseason matches from the dataset. (1)

    Average correlation between pre-season and season match outcomes is on average 0.18. It is really far away from having a correlation near -1 or 1. So pre-season means very little what comes to the regular season success. (2)

    1. https://en.wikipedia.org/wiki/Season_structure_of_the_NHL

    2. https://medium.com/@mattdesfosses/the-nhl-pre-season-does-not-predict-the-regular-season-right-d554f8d44405
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goals and Expected Goals (xG)

    ### Actual Goals vs. Stochasticity in Ice Hockey
    Ice hockey is inherently a low-scoring and high-variance sport compared to other major team sports. In the modern NHL, games average approximately 6.0 to 6.4 total goals (~3.1 goals per team). Realized goals in any single game or short sequence of games are heavily governed by random noise—colloquially termed "puck luck"—such as erratic goalie variance, minor deflections off skates, and screening effects (1, 4).

    Because raw goals have high variance and regress strongly toward the league average, relying purely on recent actual goal counts (Goals For / Goals Against) can deceive predictive models and cause overfitting on short-term noise rather than true underlying team skill.

    ### Concept of Expected Goals (xG)
    To isolate underlying team quality from finishing luck and goaltending variance, sports analytics developed **Expected Goals ($xG$)** (1, 3). 
    
    An $xG$ framework estimates the probabilistic quality of every unblocked shot attempt:
    $$xG = P(\text{Goal} \mid \mathbf{x}_{\text{shot}})$$
    where $\mathbf{x}_{\text{shot}}$ includes shot characteristics such as:
    - **Distance and shooting angle** to the center of the goal line.
    - **Shot type:** Wrist shot, slap shot, backhand, deflection/tip-in, or wrap-around.
    - **Game context:** Rebound shot (taken within 2–3 seconds of a prior save), rush shot (transition off the rush), or power play situation.
    - **Danger zone:** High-danger scoring chances (slot and crease area) vs. perimeter low-danger attempts.

    Empirical research demonstrates that shot-quality metrics and unblocked shot volumes possess vastly superior season-to-season repeatability and out-of-sample predictive power for future goal differentials and wins compared to actual past goals (2, 4).

    ### Availability in NHL API & Feature Engineering Decision
    In our project, the official NHL REST API (`api-web.nhle.com` and `api.nhle.com/stats/rest`) **does not provide pre-calculated expected goals ($xG$)**.

    Calculating full native $xG$ from scratch would require:
    1. Ingesting full play-by-play (PBP) event coordinate logs across thousands of matches (~60–70 shot events per game).
    2. Extracting exact $(x, y)$ ice coordinates, angle geometry, and temporal rebound deltas.
    3. Training and maintaining a separate shot-level machine learning model (e.g., gradient boosted trees on shot coordinates).

    **Why raw xG was omitted and how it is proxied in the Gold layer:**
    To maintain an efficient, reliable, and production-ready data pipeline without adding extreme latency and multi-tier model error propagation, we utilize validated, highly correlated proxy metrics available directly in our dbt Medallion pipeline:
    - **Fenwick % (USAT - Unblocked Shot Attempts %):** Excludes blocked shots and correlates at $r > 0.95$ with cumulative $xG$ at the team level over regular season samples (1, 4).
    - **Corsi % (SAT - Total Shot Attempts %):** Captures total territorial puck possession and offensive zone pressure.
    - **PDO (5v5 Puck Luck Metric):** Sum of shooting percentage and save percentage ($\text{Sh\%} + \text{Sv\%}$). In the long run, team PDO regresses to 1000 (100.0%). Teams with unsustainably high PDO (> 1020) are due for downward regression, while teams with low PDO (< 980) are primed for positive regression (4).
    - **Leak-Free Expanding & Rolling Goal Metrics:** Rather than raw season totals that cause look-ahead leakage, our dbt Gold layer computes strictly lagged expanding season averages (`home_in_season_goals_for_per_game`, `away_in_season_goals_for_per_game`) and short-term rolling momentum (`rolling_goals_for_5`, `rolling_goals_against_5`) over preceding games only.

    1. Macdonald B. An expected goals model for evaluating NHL teams and players. Proceedings of the 2012 MIT Sloan Sports Analytics Conference. Saatavilla: http://www.sloansportsconference.com/wp-content/uploads/2012/02/107-Expected-goals-NHL.pdf
    2. Thomas AC, Ventura SL, Jensen S, Ma HS. Competing process hazard function models for player ratings in ice hockey. The Annals of Applied Statistics. 2013;7(3):1497-1524.
    3. Schuckers M, Curro J. Total Hockey Rating (THoR): A comprehensive statistical rating of National Hockey League forwards and defensemen. MIT Sloan Sports Analytics Conference. 2013.
    4. Vollman R. Stat Shot: The Ultimate Guide to Hockey Analytics. Toronto: ECW Press; 2016.
    5. Porela M. Predicting NHL match outcomes using a data-driven model [master's thesis]. Espoo: Aalto University; 2024. Saatavilla: https://aaltodoc.aalto.fi/server/api/core/bitstreams/c3eb25e6-a001-467d-94d6-0b6fdab2f432/content
    """)
    return


if __name__ == "__main__":
    app.run()
