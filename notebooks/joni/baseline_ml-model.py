import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt


    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    from sklearn.metrics import log_loss, roc_curve

    return (
        LogisticRegression,
        OneHotEncoder,
        accuracy_score,
        alt,
        brier_score_loss,
        duckdb,
        log_loss,
        mo,
        np,
        pl,
        roc_auc_score,
        roc_curve,
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


@app.cell
def _(mo):
    mo.md(r"""
    # Baseline model, team identities and home/away information

    This is our baseline model

    ### Plan:
    1. **Target:** Y=1, home-win, Y=0, away-win
    2. **Exploratory Visualizations:** Analyze historical home-ice advantage across seasons and goal distributions.
    3. **Validation Strategy:** Set up a strict chronological train/test split (no lookahead data leakage!).
    4. **Features:** Use solely Home Team and Away Team categorical identities (one-hot encoded) to capture baseline team strength and home-ice premium.
    5. **Model & Metrics:** Train a Logistic Regression baseline and compare it against a naive class-prior baseline using **Log Loss**, **Brier Score**, **ROC AUC**, and **Accuracy**.
    6. **Diagnostic Visualizations:** Plot ROC curves and inspect model-learned team ratings.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Baseline Model: Team Identities and Home Ice

    ### Objective
    Construct a simple, transparent baseline model to predict regular season game outcomes ($$Y=1$$ for home win, $$Y=0$$ for away win) using only:
    - **Home / Away designation**
    - **Team identities**
    - **Final scores** (used strictly to define match outcomes)

    ### Steps
    1. **Data Preparation:** Extract teams and final scores; construct binary outcome and goal differentials.
    2. **Exploratory Visualizations:** Analyze home vs. away goal scoring and seasonal home ice advantage.
    3. **Chronological Splitting:** Separate earlier seasons (train) from the final season (test) to prevent lookahead bias.
    4. **Model Training:**
       - *Naive Baseline:* Predicts constant historical home win rate.
       - *Logistic Regression:* Estimates team strengths and home ice advantage via one-hot encoded team identities.
    5. **Evaluation:** Compare probabilistic predictions using Log Loss, Brier Score, ROC AUC, and Accuracy.
    6. **Model Diagnostics:** Plot ROC curves and inspect model-estimated team strength rankings.
    """)
    return


@app.cell
def _(filtered_games, pl):
    # Step 1: Prepare modeling dataset using Polars
    bl_home_team_col = next((c for c in ["home_team_name", "home_team_abbrev", "home_team_common_name", "home_team", "home_team_id"] if c in filtered_games.columns), "home_team_id")
    bl_away_team_col = next((c for c in ["away_team_name", "away_team_abbrev", "away_team_common_name", "away_team", "away_team_id"] if c in filtered_games.columns), "away_team_id")
    bl_home_score_col = next((c for c in ["home_score", "home_team_score", "home_goals", "home_total_goals"] if c in filtered_games.columns), "home_score")
    bl_away_score_col = next((c for c in ["away_score", "away_team_score", "away_goals", "away_total_goals"] if c in filtered_games.columns), "away_score")

    bl_data = (
        filtered_games
        .select([
            pl.col("game_id"),
            pl.col("season").cast(pl.Utf8),
            pl.col("game_date"),
            pl.col(bl_home_team_col).cast(pl.Utf8).alias("home_team"),
            pl.col(bl_away_team_col).cast(pl.Utf8).alias("away_team"),
            pl.col(bl_home_score_col).cast(pl.Int32).alias("home_score"),
            pl.col(bl_away_score_col).cast(pl.Int32).alias("away_score"),
        ])
        .filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
        .with_columns([
            (pl.col("home_score") > pl.col("away_score")).cast(pl.Int32).alias("home_win"),
            (pl.col("home_score") - pl.col("away_score")).alias("goal_differential"),
        ])
        .sort("game_date")
    )

    bl_summary = bl_data.select([
        pl.len().alias("total_games"),
        pl.col("home_team").n_unique().alias("unique_teams"),
        pl.col("season").n_unique().alias("num_seasons"),
        (pl.col("home_win").mean() * 100).round(2).alias("home_win_pct"),
        pl.col("home_score").mean().round(2).alias("avg_home_score"),
        pl.col("away_score").mean().round(2).alias("avg_away_score"),
    ])

    bl_summary
    return (bl_data,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exploratory Visualizations: Scoring & Home Ice Advantage

    We examine:
    1. **Goal Distribution Comparison:** Frequency of goals scored by home teams versus away teams.
    2. **Seasonal Home Advantage:** Proportion of home victories across seasons compared to the 50% parity baseline ($$p=0.50$$).
    """)
    return


@app.cell
def _(alt, bl_data, pl):
    # Reshape goals for a side-by-side distribution chart
    bl_goals_long = (
        bl_data
        .select(["game_id", "home_score", "away_score"])
        .unpivot(index="game_id", on=["home_score", "away_score"], variable_name="location", value_name="goals")
        .with_columns(
            pl.when(pl.col("location") == "home_score")
            .then(pl.lit("Home"))
            .otherwise(pl.lit("Away"))
            .alias("location")
        )
        .group_by(["goals", "location"])
        .agg(pl.len().alias("count"))
        .sort(["goals", "location"])
    )

    bl_goal_dist_chart = (
        alt.Chart(bl_goals_long)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("goals:O", title="Goals Scored per Game"),
            y=alt.Y("count:Q", title="Frequency", stack=None),
            color=alt.Color("location:N", title="Location", scale=alt.Scale(domain=["Home", "Away"], range=["#1f77b4", "#ff7f0e"])),
            tooltip=[
                alt.Tooltip("location:N", title="Location"),
                alt.Tooltip("goals:O", title="Goals"),
                alt.Tooltip("count:Q", title="Total Games"),
            ],
        )
        .properties(
            title="Home vs. Away Goals Scored Distribution",
            width=320,
            height=280,
        )
    )

    # Seasonal home win rate chart
    bl_season_rates = (
        bl_data
        .group_by("season")
        .agg([
            pl.len().alias("games"),
            pl.col("home_win").mean().alias("win_rate"),
            (pl.col("home_win").mean() * 100).round(2).alias("win_pct"),
        ])
        .sort("season")
    )

    bl_season_chart = (
        alt.Chart(bl_season_rates)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#2b5c8f")
        .encode(
            x=alt.X("season:N", title="Season", sort=None),
            y=alt.Y("win_rate:Q", title="Home Win Rate", scale=alt.Scale(domain=[0.4, 0.65]), axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("season:N", title="Season"),
                alt.Tooltip("games:Q", title="Games"),
                alt.Tooltip("win_pct:Q", title="Home Win %", format=".2f"),
            ],
        )
        .properties(
            title="Home Win Rate by Season",
            width=320,
            height=280,
        )
    )

    bl_parity_line = (
        alt.Chart(pl.DataFrame({"y": [0.5]}))
        .mark_rule(strokeDash=[4, 4], color="crimson", strokeWidth=1.5)
        .encode(y="y:Q")
    )

    alt.hconcat(bl_goal_dist_chart, bl_season_chart + bl_parity_line)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Chronological Train / Test Split

    To avoid temporal data leakage (training on future matches to predict past matches), we partition games chronologically:
    - **Training Set:** All seasons prior to the latest season.
    - **Test Set:** The latest season in the dataset.
    """)
    return


@app.cell
def _(bl_data, pl):
    # Split chronologically
    bl_seasons = sorted(bl_data["season"].unique().to_list())

    if len(bl_seasons) > 1:
        bl_test_season_name = bl_seasons[-1]
        bl_train_season_names = bl_seasons[:-1]
        bl_train = bl_data.filter(pl.col("season").is_in(bl_train_season_names))
        bl_test = bl_data.filter(pl.col("season") == bl_test_season_name)
    else:
        bl_split_point = int(len(bl_data) * 0.8)
        bl_train = bl_data.slice(0, bl_split_point)
        bl_test = bl_data.slice(bl_split_point)
        bl_test_season_name = "Holdout 20%"
        bl_train_season_names = ["Historical 80%"]

    bl_split_summary = pl.DataFrame({
        "Dataset": ["Train Set", "Test Set"],
        "Season(s)": [", ".join(bl_train_season_names), str(bl_test_season_name)],
        "Total Games": [len(bl_train), len(bl_test)],
        "Home Win Rate (%)": [
            round(float(bl_train["home_win"].mean() * 100), 2),
            round(float(bl_test["home_win"].mean() * 100), 2),
        ],
    })

    bl_split_summary
    return bl_test, bl_train


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Formulation

    We compare two models:

    1. **Naive Class Prior:**
       Predicts a constant probability equal to the historical training home win rate:
       $$\hat{p} = \bar{Y}_{\text{train}}$$

    2. **Logistic Regression (Team Identities):**
       One-hot encodes the home team and away team identifiers:
       $$P(\text{Home Win}) = \sigma\left(\beta_0 + \sum_{j} w_{h,j} \cdot \text{Home}_j + \sum_{j} w_{a,j} \cdot \text{Away}_j\right)$$
       - Intercept $$\beta_0$$ captures global league-wide home advantage.
       - Coefficients $$w_{h,j}$$ and $$w_{a,j}$$ capture team-specific performance at home and away.
    """)
    return


@app.cell
def _(LogisticRegression, OneHotEncoder, bl_test, bl_train, np):
    # Feature encoding and model fitting
    bl_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    bl_X_train = bl_encoder.fit_transform(bl_train.select(["home_team", "away_team"]).to_numpy())
    bl_X_test = bl_encoder.transform(bl_test.select(["home_team", "away_team"]).to_numpy())

    bl_y_train = bl_train["home_win"].to_numpy()
    bl_y_test = bl_test["home_win"].to_numpy()

    # 1. Naive baseline predictions
    bl_prior_prob = float(bl_y_train.mean())
    bl_preds_naive_prob = np.full_like(bl_y_test, fill_value=bl_prior_prob, dtype=float)
    bl_preds_naive_class = (bl_preds_naive_prob >= 0.5).astype(int)

    # 2. Logistic Regression baseline
    bl_model_lr = LogisticRegression(C=0.1, solver="liblinear", random_state=42)
    bl_model_lr.fit(bl_X_train, bl_y_train)

    bl_preds_lr_prob = bl_model_lr.predict_proba(bl_X_test)[:, 1]
    bl_preds_lr_class = (bl_preds_lr_prob >= 0.5).astype(int)
    return (
        bl_encoder,
        bl_model_lr,
        bl_preds_lr_class,
        bl_preds_lr_prob,
        bl_preds_naive_class,
        bl_preds_naive_prob,
        bl_y_test,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Evaluation

    Performance is assessed using probabilistic and classification metrics:
    - **Log Loss:** Penalizes confident incorrect probabilistic predictions ($$-\frac{1}{N}\sum [y\log p + (1-y)\log(1-p)]$$).
    - **Brier Score:** Mean squared error of predicted probabilities ($$\frac{1}{N}\sum (p - y)^2$$).
    - **ROC AUC:** Discrimination ability across all classification thresholds.
    - **Accuracy:** Proportion of correct discrete predictions at threshold $$p \ge 0.5$$.
    """)
    return


@app.cell
def _(
    accuracy_score,
    bl_preds_lr_class,
    bl_preds_lr_prob,
    bl_preds_naive_class,
    bl_preds_naive_prob,
    bl_y_test,
    brier_score_loss,
    log_loss,
    pl,
    roc_auc_score,
):
    # Evaluation table
    bl_results_table = pl.DataFrame({
        "Model": ["Naive Prior", "Logistic Regression (Teams + Home)"],
        "Log Loss (lower is better)": [
            round(log_loss(bl_y_test, bl_preds_naive_prob), 4),
            round(log_loss(bl_y_test, bl_preds_lr_prob), 4),
        ],
        "Brier Score (lower is better)": [
            round(brier_score_loss(bl_y_test, bl_preds_naive_prob), 4),
            round(brier_score_loss(bl_y_test, bl_preds_lr_prob), 4),
        ],
        "ROC AUC (higher is better)": [
            round(roc_auc_score(bl_y_test, bl_preds_naive_prob), 4),
            round(roc_auc_score(bl_y_test, bl_preds_lr_prob), 4),
        ],
        "Accuracy (%)": [
            round(accuracy_score(bl_y_test, bl_preds_naive_class) * 100, 2),
            round(accuracy_score(bl_y_test, bl_preds_lr_class) * 100, 2),
        ],
    })

    bl_results_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Diagnostics: ROC Curve & Team Net Strength

    1. **ROC Curve:** Measures sensitivity against false positive rate across probability thresholds.
    2. **Team Net Rating:** Computed from logistic regression weights as:
       $$\text{Rating}_{\text{team}} = w_{\text{home}} - w_{\text{away}}$$
       Positive ratings reflect stronger teams relative to the league average.
    """)
    return


@app.cell
def _(
    alt,
    bl_encoder,
    bl_model_lr,
    bl_preds_lr_prob,
    bl_y_test,
    pl,
    roc_auc_score,
    roc_curve,
):
    # Diagnostic Visualizations
    # 1. ROC Curve
    bl_fpr, bl_tpr, _ = roc_curve(bl_y_test, bl_preds_lr_prob)
    bl_auc = roc_auc_score(bl_y_test, bl_preds_lr_prob)

    bl_roc_df = pl.DataFrame({"fpr": bl_fpr, "tpr": bl_tpr})
    bl_diag_df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})

    bl_roc_chart = (
        alt.Chart(bl_roc_df)
        .mark_line(color="#2ca02c", strokeWidth=2.5)
        .encode(
            x=alt.X("fpr:Q", title="False Positive Rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="True Positive Rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("fpr:Q", title="FPR", format=".3f"),
                alt.Tooltip("tpr:Q", title="TPR", format=".3f"),
            ],
        )
        .properties(
            title=f"Test ROC Curve (AUC = {bl_auc:.3f})",
            width=320,
            height=300,
        )
    )

    bl_diag_chart = (
        alt.Chart(bl_diag_df)
        .mark_line(strokeDash=[4, 4], color="gray", strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q")
    )

    # 2. Team Net Strength
    bl_names = bl_encoder.get_feature_names_out(["home", "away"])
    bl_weights = bl_model_lr.coef_[0]

    bl_strength_records = []
    for bl_i, bl_name in enumerate(bl_names):
        if bl_name.startswith("home_"):
            bl_t = bl_name.replace("home_", "")
            bl_away_name = f"away_{bl_t}"
            bl_away_idx = next((j for j, n in enumerate(bl_names) if n == bl_away_name), None)
            bl_w_home = bl_weights[bl_i]
            bl_w_away = bl_weights[bl_away_idx] if bl_away_idx is not None else 0.0
            bl_strength_records.append({
                "team": bl_t,
                "net_rating": float(bl_w_home - bl_w_away),
            })

    bl_ratings_chart = (
        alt.Chart(
            pl.DataFrame(bl_strength_records)
            .sort("net_rating", descending=True)
            .head(12)
        )
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, color="#1f77b4")
        .encode(
            x=alt.X("net_rating:Q", title="Net Rating (Weight Differential)"),
            y=alt.Y("team:N", title="Team", sort="-x"),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("net_rating:Q", title="Net Rating", format=".3f"),
            ],
        )
        .properties(
            title="Top 12 Teams by Estimated Net Rating",
            width=320,
            height=300,
        )
    )

    alt.hconcat(bl_roc_chart + bl_diag_chart, bl_ratings_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # outcome

    Whole reason to make this simple model is to test our pipeline, features were selected before hand so there were no feature engineering made. Everything moves smoothly to the notebooks and EDA revealed more information of the data that we knew earlier.

    Compared to our Naive prior that predicted always home win our logistic regression model learned to compare teams which one is stronger and take consider also the home advance. (RCO AUC 0.500 to 0.597)

    Homewinrates were different on train and test sets and that is the reason why accuracy is the same in both Naive and Logistics regression.
    """)
    return


if __name__ == "__main__":
    app.run()
