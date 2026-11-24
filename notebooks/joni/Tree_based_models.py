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

    from sklearn.preprocessing import OrdinalEncoder

    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    from sklearn.metrics import log_loss, roc_curve
    from sklearn.inspection import permutation_importance

    from sklearn.impute import SimpleImputer

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    import xgboost as xgb
    import catboost as cb

    return (
        DecisionTreeClassifier,
        HistGradientBoostingClassifier,
        OrdinalEncoder,
        RandomForestClassifier,
        SimpleImputer,
        accuracy_score,
        alt,
        brier_score_loss,
        cb,
        dt,
        duckdb,
        log_loss,
        mo,
        np,
        permutation_importance,
        pl,
        roc_auc_score,
        roc_curve,
        xgb,
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


@app.cell
def _(RandomForestClassifier, X_train_imp, X_val_imp, y_train):
    #init and train randomForest

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf_model.fit(X_train_imp, y_train)

    # pred against validation

    y_val_prob_rf = rf_model.predict_proba(X_val_imp)[:, 1]
    y_val_pred_rf = rf_model.predict(X_val_imp)
    return y_val_pred_rf, y_val_prob_rf


@app.cell
def _(
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    y_val,
    y_val_pred_naive,
    y_val_pred_rf,
    y_val_prob_rf,
    y_val_prob_tree,
):
    print("Random Forest (100 trees, depth=4)")
    print(f"Accuracy:    {accuracy_score(y_val, y_val_pred_rf)}  (Naive: {accuracy_score(y_val, y_val_pred_naive)})")
    print(f"ROC-AUC:     {roc_auc_score(y_val, y_val_prob_rf)}  (Single Tree: {roc_auc_score(y_val, y_val_prob_tree)})")
    print(f"Brier Score: {brier_score_loss(y_val, y_val_prob_rf)}  (Single Tree: {brier_score_loss(y_val, y_val_prob_tree)})")
    print(f"Log Loss:    {log_loss(y_val, y_val_prob_rf)}  (Single Tree: {log_loss(y_val, y_val_prob_tree)})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    LightGBM training
    """)
    return


@app.cell
def _(HistGradientBoostingClassifier, X_train, X_val, y_train):
    #init model
    # max_iter = amount of trees
    # max_depth = As seen, lets keep trees low so test with 2 to 4

    hgb_model = HistGradientBoostingClassifier(
        max_iter = 100,
        max_depth = 1,
        random_state = 42
    )

    # Lets train it without imputer
    hgb_model.fit(X_train, y_train)

    # Predict with validation data

    y_val_prob_hgb = hgb_model.predict_proba(X_val)[:,1]
    y_val_pred_hgb = hgb_model.predict(X_val)
    return hgb_model, y_val_pred_hgb, y_val_prob_hgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tried getting better results altering learning_rate and min_samples_leaf, but got best results without them
    """)
    return


@app.cell
def _(
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    y_val,
    y_val_pred_hgb,
    y_val_pred_rf,
    y_val_prob_hgb,
    y_val_prob_rf,
):
    print("HistGradientBoosting vs. earlier. Best trees had depth of 1")
    print(f"Accuracy:    {accuracy_score(y_val, y_val_pred_hgb)}  (RF: {accuracy_score(y_val, y_val_pred_rf)})")
    print(f"ROC-AUC:     {roc_auc_score(y_val, y_val_prob_hgb)}  (RF: {roc_auc_score(y_val, y_val_prob_rf)})")
    print(f"Brier Score: {brier_score_loss(y_val, y_val_prob_hgb)}  (RF: {brier_score_loss(y_val, y_val_prob_rf)})")
    print(f"Log Loss:    {log_loss(y_val, y_val_prob_hgb)}  (RF: {log_loss(y_val, y_val_prob_rf)})")
    return


@app.cell
def _(X_val, feature_cols, hgb_model, permutation_importance, pl, y_val):
    # Calculate permutation importance in validation data

    perm_importance = permutation_importance(
        hgb_model, 
        X_val, 
        y_val, 
        n_repeats=10, 
        random_state=42, 
        scoring="roc_auc"
    )

    #Lets make Polars DataGrame from the resulta
    importance_df = pl.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort("importance_mean", descending=True)

    # top10
    importance_df.head(10)
    return


@app.cell
def _(
    roc_auc_score,
    roc_curve,
    y_val,
    y_val_prob_hgb,
    y_val_prob_rf,
    y_val_prob_tree,
):
    # 1. Decision Tree
    auc_tree = roc_auc_score(y_val, y_val_prob_tree)
    fpr_tree, tpr_tree, _ = roc_curve(y_val, y_val_prob_tree)

    # 2. Random Forest
    auc_rf = roc_auc_score(y_val, y_val_prob_rf)
    fpr_rf, tpr_rf, _ = roc_curve(y_val, y_val_prob_rf)

    # 3. HistGradientBoosting 
    auc_hgb = roc_auc_score(y_val, y_val_prob_hgb)
    fpr_hgb, tpr_hgb, _ = roc_curve(y_val, y_val_prob_hgb)
    return


@app.cell
def _(X_train, X_val, cb, xgb, y_train):
    # 1. XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=1,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)

    # 2. CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=100,
        depth=1,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(X_train, y_train)

    # 3. predict against validation data
    y_val_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
    y_val_pred_xgb = xgb_model.predict(X_val)

    y_val_prob_cb = cb_model.predict_proba(X_val)[:, 1]
    y_val_pred_cb = cb_model.predict(X_val)
    return y_val_pred_cb, y_val_pred_xgb, y_val_prob_cb, y_val_prob_xgb


@app.cell
def _(
    alt,
    pl,
    roc_auc_score,
    roc_curve,
    y_val,
    y_val_prob_cb,
    y_val_prob_hgb,
    y_val_prob_rf,
    y_val_prob_tree,
    y_val_prob_xgb,
):
    # 1. Decision Tree pisteet
    _auc_tree = roc_auc_score(y_val, y_val_prob_tree)
    _fpr_tree, _tpr_tree, _ = roc_curve(y_val, y_val_prob_tree)

    # 2. Random Forest pisteet
    _auc_rf = roc_auc_score(y_val, y_val_prob_rf)
    _fpr_rf, _tpr_rf, _ = roc_curve(y_val, y_val_prob_rf)

    # 3. HistGradientBoosting pisteet
    _auc_hgb = roc_auc_score(y_val, y_val_prob_hgb)
    _fpr_hgb, _tpr_hgb, _ = roc_curve(y_val, y_val_prob_hgb)

    # 4. XGBoost pisteet
    _auc_xgb = roc_auc_score(y_val, y_val_prob_xgb)
    _fpr_xgb, _tpr_xgb, _ = roc_curve(y_val, y_val_prob_xgb)

    # 5. CatBoost pisteet
    _auc_cb = roc_auc_score(y_val, y_val_prob_cb)
    _fpr_cb, _tpr_cb, _ = roc_curve(y_val, y_val_prob_cb)

    # Yhdistetään kaikki 5 mallia yhteen taulukkoon
    _roc_all_df = pl.concat([
        pl.DataFrame({
            "fpr": _fpr_tree,
            "tpr": _tpr_tree,
            "Model": f"Decision Tree (AUC = {_auc_tree:.3f})"
        }),
        pl.DataFrame({
            "fpr": _fpr_rf,
            "tpr": _tpr_rf,
            "Model": f"Random Forest (AUC = {_auc_rf:.3f})"
        }),
        pl.DataFrame({
            "fpr": _fpr_hgb,
            "tpr": _tpr_hgb,
            "Model": f"HistGradientBoosting (AUC = {_auc_hgb:.3f})"
        }),
        pl.DataFrame({
            "fpr": _fpr_xgb,
            "tpr": _tpr_xgb,
            "Model": f"XGBoost (AUC = {_auc_xgb:.3f})"
        }),
        pl.DataFrame({
            "fpr": _fpr_cb,
            "tpr": _tpr_cb,
            "Model": f"CatBoost (AUC = {_auc_cb:.3f})"
        })
    ])

    # Satunnaisarvauksen diagonaaliviiva
    _diagonal_df = pl.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})

    # Kaikkien mallien käyrät
    _chart_models = alt.Chart(_roc_all_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("fpr:Q", title="False Positive Rate", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("tpr:Q", title="True Positive Rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", title="Model")
    )

    # Coinflip 
    _chart_diagonal = alt.Chart(_diagonal_df).mark_line(
        strokeDash=[5, 5], 
        color="gray", 
        strokeWidth=1.5
    ).encode(
        x="fpr:Q", 
        y="tpr:Q"
    )

    # Lets add all charts together
    all_roc_chart = (_chart_diagonal + _chart_models).properties(
        title="Kaikkien mallien ROC-vertailu validointidatalla",
        width=550,
        height=400
    )

    all_roc_chart
    return


@app.cell
def _(
    accuracy_score,
    brier_score_loss,
    log_loss,
    pl,
    roc_auc_score,
    y_val,
    y_val_pred_cb,
    y_val_pred_hgb,
    y_val_pred_naive,
    y_val_pred_rf,
    y_val_pred_tree,
    y_val_pred_xgb,
    y_val_prob_cb,
    y_val_prob_hgb,
    y_val_prob_naive,
    y_val_prob_rf,
    y_val_prob_tree,
    y_val_prob_xgb,
):
    comparison_df = pl.DataFrame({
        "Malli": [
            "Naive Baseline",
            "Decision Tree (depth=4)",
            "Random Forest (100 trees)",
            "HistGradientBoosting",
            "XGBoost",
            "CatBoost"
        ],
        "ROC-AUC": [
            roc_auc_score(y_val, y_val_prob_naive),
            roc_auc_score(y_val, y_val_prob_tree),
            roc_auc_score(y_val, y_val_prob_rf),
            roc_auc_score(y_val, y_val_prob_hgb),
            roc_auc_score(y_val, y_val_prob_xgb),
            roc_auc_score(y_val, y_val_prob_cb),
        ],
        "Log Loss": [
            log_loss(y_val, y_val_prob_naive),
            log_loss(y_val, y_val_prob_tree),
            log_loss(y_val, y_val_prob_rf),
            log_loss(y_val, y_val_prob_hgb),
            log_loss(y_val, y_val_prob_xgb),
            log_loss(y_val, y_val_prob_cb),
        ],
        "Brier Score": [
            brier_score_loss(y_val, y_val_prob_naive),
            brier_score_loss(y_val, y_val_prob_tree),
            brier_score_loss(y_val, y_val_prob_rf),
            brier_score_loss(y_val, y_val_prob_hgb),
            brier_score_loss(y_val, y_val_prob_xgb),
            brier_score_loss(y_val, y_val_prob_cb),
        ],
        "Accuracy": [
            accuracy_score(y_val, y_val_pred_naive),
            accuracy_score(y_val, y_val_pred_tree),
            accuracy_score(y_val, y_val_pred_rf),
            accuracy_score(y_val, y_val_pred_hgb),
            accuracy_score(y_val, y_val_pred_xgb),
            accuracy_score(y_val, y_val_pred_cb),
        ]
    }).sort("ROC-AUC", descending=True)

    comparison_df
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extra study hypothesis
    Earlier baseline model learned alot just getting team names, lets test how much it helps model to predict.

    Other hypothesis is that SimpleImpute is too simple and for the first 10 games it evens the teams too much. Lets find algorithm to fill the Nulls with better accuracy base on studies. So how close Decision tree gets to LightGBM.
    """)
    return


@app.cell
def _(
    OrdinalEncoder,
    X_train,
    X_val,
    feature_cols,
    pl,
    train_df,
    validation_df,
):
    # Lets add team names to feature cols
    feature_and_team_cols = feature_cols + ["home_team_abbrev", "away_team_abbrev"]

    # 2. Lets take data
    X_train_teams = train_df.select(feature_and_team_cols)
    X_val_teams = validation_df.select(feature_and_team_cols)

    # encode team name to numbers
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    teams_train_encoded = encoder.fit_transform(X_train_teams.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())
    teams_val_encoded = encoder.transform(X_val_teams.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())

    # connect encoded team names to DataFrame
    X_train_with_teams = X_train.with_columns([
        pl.Series("home_team_encoded", teams_train_encoded[:, 0]),
        pl.Series("away_team_encoded", teams_train_encoded[:, 1])
    ])

    X_val_with_teams = X_val.with_columns([
        pl.Series("home_team_encoded", teams_val_encoded[:, 0]),
        pl.Series("away_team_encoded", teams_val_encoded[:, 1])
    ])
    return X_train_with_teams, X_val_with_teams


@app.cell
def _(
    HistGradientBoostingClassifier,
    X_train_with_teams,
    X_val_with_teams,
    y_train,
):
    # Lets tell that two last cols are categorical
    cat_indices = [
        X_train_with_teams.columns.index("home_team_encoded"),
        X_train_with_teams.columns.index("away_team_encoded")
    ]

    hgb_teams_model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=1,
        categorical_features=cat_indices,
        random_state=42
    )

    # Lets train
    hgb_teams_model.fit(X_train_with_teams, y_train)

    # Predict
    y_val_prob_hgb_teams = hgb_teams_model.predict_proba(X_val_with_teams)[:, 1]
    y_val_pred_hgb_teams = hgb_teams_model.predict(X_val_with_teams)
    return (y_val_prob_hgb_teams,)


@app.cell
def _(log_loss, roc_auc_score, y_val, y_val_prob_hgb, y_val_prob_hgb_teams):
    print("stats vs. stats + team names")
    print(f"ROC-AUC without names: {roc_auc_score(y_val, y_val_prob_hgb)}")
    print(f"ROC-AUC with names: {roc_auc_score(y_val, y_val_prob_hgb_teams)}")
    print(f"Log Loss without names: {log_loss(y_val, y_val_prob_hgb)}")
    print(f"Log Loss with names: {log_loss(y_val, y_val_prob_hgb_teams)}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
