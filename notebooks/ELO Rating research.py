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
    # Elo rating - FiveThirtyEight's model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    FiveThirtyEight model has been used to predict elections in USA. (1) Its also useful for making ELO-rating for NHL teams with little tuning. Where home advantage is calculated 50 to 42 and K-factor is increased from 6 to 8. At the start of the season model adds Vegas totals bias for team starting ELO. (2)

    ELO-rating of all historical games can be found in one csv that starts in year 2018.

    Rating can be used to predict proba of the match win:
    $$E_A = \frac{1}{1 + 10^{(R_B - (R_A + H)) / 400}}$$

    Usually teams start with 1500 or other ELO-rating or as we have multiple season data, between season the ELO is deprecated to take consider teams change between seasons.

    $R_{\text{new}} = 0.75 \cdot R_{\text{old}} + 0.25 \cdot 1500$

    How the ELO-rating is calculated after the match.

    $$R'_A = R_A + K \cdot (S_A - E_A)$$

    $$R'_B = R_B - K \cdot (S_A - E_A)$$

    Tuning of the model is based on K-multiplier, which is increased from 6 to 8. Multiplier can be also added to K from margin of victory, to reward high goal difference.

    We need to be really careful of future data leakage when applying ELO-rating based attributes.

    ELO-rating can be used in different ways as a feature. *Elo-rating* as standard baseline feature. Also we can make a *ELO-diff* between teams, that for tree models is strong feature to choose from. *ELO-win proba*, gives us precise win proba. *ELO-momentum* gives us signal with rolling numbers how the team is doing in short term.

    Pitfalls

    Greedy XGBoost might pass on other signals if we use ELO-diff, because its to dominating feature gives most loss reduction. impurity-based feature importance for trees is strongly biased and favor high cardinality features (4). colsample_bytree / colsample_bylevel / colsample_bynode hyperparameter can fix this, needs to be tested with lower than default 1.0 value (5). Instead of ELO-diff, we can use elo-proba and turn it into log-odds and feed it to the model.

    XGBoost: xgb.DMatrix(X, label=y, base_margin=elo_log_odds)

    CatBoost: Pool(X, y, baseline=elo_log_odds)

    $$\text{base\_margin} = \ln\left(\frac{E_A}{1 - E_A}\right)$$

    1. https://en.wikipedia.org/wiki/FiveThirtyEight
    2. https://elo.harvitronix.com/about
    3. https://github.com/fivethirtyeight/data/tree/master/nhl-forecasts
    4. https://scikit-learn.org/stable/modules/permutation_importance.html
    5. https://xgboost.readthedocs.io/en/stable/parameter.html
    """)
    return


if __name__ == "__main__":
    app.run()
