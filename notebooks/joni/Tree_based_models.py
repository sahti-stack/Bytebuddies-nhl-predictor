import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import duckdb
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    import datetime as dt

    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    from sklearn.metrics import log_loss, roc_curve

    from sklearn.impute import SimpleImputer

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    return (
        DecisionTreeClassifier,
        SimpleImputer,
        accuracy_score,
        alt,
        brier_score_loss,
        dt,
        duckdb,
        log_loss,
        mo,
        np,
        pl,
        roc_auc_score,
    )


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
        WHERE
        season != 20212022
        AND
        game_type == 2
        AND
        is_played == True
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
def _(df, pl):
    # Check that UTA prior stats are non-nulls

    df.filter(
        (pl.col("season") == 20242025) & ((pl.col("home_team_abbrev") == "UTA") | (pl.col("away_team_abbrev") == "UTA"))
    ).select([
        "game_date", "home_team_abbrev", "away_team_abbrev", 
        "home_prior_point_pct", "away_prior_point_pct"
    ])
    return


@app.cell
def _(df, pl):
    # What colls still has nulls, if no prior is null our burn in season works
    df_clean = df.filter(pl.col("season") != 20212022)

    [col for col in df_clean.columns if df_clean[col].null_count() > 0]
    # Last nulls are start of the season, need to think what we are going to do to them
    return (df_clean,)


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


@app.cell
def _(df_clean):
    #group features, cols for information and target col
    target_col = "home_win"

    post_game_cols = [
        "home_score", "away_score", "total_goals", "goal_diff", "away_win", "is_regulation", "is_overtime"
    ]

    metadata_cols = [
        "game_id", "season", "game_type", "game_date", "start_time_utc", "venue", "game_state", "is_played"
    ] 

    team_cols = [
        "home_team_abbrev", "away_team_abbrev", "home_team_id", "away_team_id", "home_team_name", "away_team_name", "home_lineage_id", "away_lineage_id"
    ]

    #excluded cols packed together

    excluded_cols = set([target_col] + post_game_cols + metadata_cols + team_cols)

    # Features are what is left

    feature_cols = [col for col in df_clean.columns if col not in excluded_cols]

    len(feature_cols)
    return feature_cols, target_col


@app.cell
def _(df_clean, dt, pl):
    # Lets split data
    #- Train data all games prior 30.11.2024 (season 22-23, 23-24 and start of 24-25)
    #- validation data 1.12.2024-28.2.2025
    #test data 1.3.2025-
    # To split we use game_date "datetime2024-11-30"

    #train data
    train_df = df_clean.filter(pl.col("game_date") <= dt.datetime(2024, 11, 30))

    #validation data
    validation_df = df_clean.filter(
        (pl.col("game_date") >= dt.datetime(2024, 12, 1)) & (pl.col("game_date") <= dt.datetime(2025,2,28)))

    # test data
    test_df = df_clean.filter(pl.col("game_date") >= dt.datetime(2025,3,1))
    return test_df, train_df, validation_df


@app.cell
def _(feature_cols, target_col, test_df, train_df, validation_df):
    # Split train, validation and test data
    X_train = train_df.select(feature_cols)
    y_train = train_df.select(target_col).to_series()

    X_val = validation_df.select(feature_cols)
    y_val = validation_df.select(target_col).to_series()

    X_test = test_df.select(feature_cols)
    y_test = test_df.select(target_col).to_series()
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _(test_df, train_df, validation_df):
    # Lets check rows on sets

    print(f"Train data matches: {len(train_df)}")
    print(f"validation matches: {len(validation_df)}")
    print(f"test matches: {len(test_df)}")
    print(f"Total matches: {len(train_df) + len(validation_df) + len(test_df)}")
    return


@app.cell
def _(y_test, y_train, y_val):
    # Lets check the home win percentage, that is our baseline
    print(f"home win mean trainset: {y_train.mean()}")
    print(f"home win mean vaildation set: {y_val.mean()}")
    print(f"home win mean test set: {y_test.mean()}")

    return


@app.cell
def _(alt, df_clean, feature_cols, pl):
    bins = 30

    # Pre-compute histogram bins and counts in Polars
    hist_df = pl.concat([
        df_clean.select(pl.col(col).drop_nulls().alias("val"))
        .with_columns(
            bin_idx=(
                ((pl.col("val") - pl.col("val").min()) / (pl.col("val").max() - pl.col("val").min()) * bins)
                .floor()
                .clip(0, bins - 1)
            ),
            min_val=pl.col("val").min(),
            max_val=pl.col("val").max(),
        )
        .with_columns(
            step=(pl.col("max_val") - pl.col("min_val")) / bins
        )
        .group_by("bin_idx", "min_val", "step")
        .agg(count=pl.len())
        .with_columns(
            feature=pl.lit(col),
            x0=pl.col("min_val") + pl.col("bin_idx") * pl.col("step"),
            x1=pl.col("min_val") + (pl.col("bin_idx") + 1) * pl.col("step"),
        )
        .select("feature", "x0", "x1", "count")
        for col in feature_cols
    ])

    chart = (
        alt.Chart(hist_df)
        .mark_bar()
        .encode(
            x=alt.X("x0:Q", bin="binned", title="Value"),
            x2="x1:Q",
            y=alt.Y("count:Q", title="Count"),
            facet=alt.Facet("feature:N", columns=3, title=None),
        )
        .properties(width=200, height=150)
        .resolve_scale(x="independent", y="independent")
    )

    chart
    return


@app.cell
def _(
    accuracy_score,
    brier_score_loss,
    log_loss,
    np,
    roc_auc_score,
    y_train,
    y_val,
):
    # lets calculate historical probability of home win
    p_naive = float(y_train.mean())

    # Vectors len of validation group

    y_val_prob_naive = np.full(len(y_val), fill_value=p_naive)

    # hard coded prob, always 1 so always home win
    y_val_pred_naive = np.ones(len(y_val))

    # Metrics for naive

    print("metrics for naive baseline with validation data")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred_naive)}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_val_prob_naive)}")
    print(f"Brier Score(MSE): {brier_score_loss(y_val, y_val_prob_naive)}")
    print(f"Log loss: {log_loss(y_val, y_val_prob_naive)}")
    return y_val_pred_naive, y_val_prob_naive


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Decision tree and random forest
    """)
    return


@app.cell
def _(SimpleImputer, X_test, X_train, X_val):
    # Start of the season have NULL values on rolling 5/10 match data, lets use SimpleImputer to fill median values of training data. We will fill train, validation and test data with same values.

    # Lets make imputer and fit it to training data only
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    # transform all three dataset with dame values

    X_train_imp = imputer.transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    return X_test_imp, X_train_imp, X_val_imp


@app.cell
def _(X_test_imp, X_train_imp, X_val_imp, np):
    print("Null values in X_train_imp: ", np.isnan(X_train_imp).sum())

    print("Null values in X_train_imp: ", np.isnan(X_val_imp).sum())

    print("Null values in X_train_imp: ", np.isnan(X_test_imp).sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Im aware that using SimpleImputer we are missing some information/signal on the start of the season with SimpleImputer and simple DecisionTreeClassifier. More modern LightGBM for example can route NULLS to use right information rather than median. This will be good lesson to see difference in results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Depth of the tree on DecisionTree and RandomForest have been in other studies most optimal with 2 and 1 "max_depth".

    https://github.com/UBC-MDS-2018-19/nhl-game-predictor/blob/master/doc/results_report.md
    """)
    return


@app.cell
def _(DecisionTreeClassifier, X_train_imp, X_val_imp, y_train):
    #init and train the tree

    tree_model = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_model.fit(X_train_imp, y_train)

    #Lets predict with our validation data for a homewin

    y_val_prob_tree = tree_model.predict_proba(X_val_imp)[:,1]
    y_val_pred_tree = tree_model.predict(X_val_imp)

    return y_val_pred_tree, y_val_prob_tree


@app.cell
def _(
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    y_val,
    y_val_pred_naive,
    y_val_pred_tree,
    y_val_prob_naive,
    y_val_prob_tree,
):
    print("Decision Tree vs. Naive Baseline")
    print(f"Accuracy:    {accuracy_score(y_val, y_val_pred_tree)}  (Naive: {accuracy_score(y_val, y_val_pred_naive)})")
    print(f"ROC-AUC:     {roc_auc_score(y_val, y_val_prob_tree)}  (Naive: 0.5000)")
    print(f"Brier Score: {brier_score_loss(y_val, y_val_prob_tree)}  (Naive: {brier_score_loss(y_val, y_val_prob_naive)})")
    print(f"Log Loss:    {log_loss(y_val, y_val_prob_tree)}  (Naive: {log_loss(y_val, y_val_prob_naive)})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tested with depth of 1-4 and as studied said, "max_depth=2" gave us best result
    """)
    return


app._unparsable_cell(
    r"""
    #init and train randomForest

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=)
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
