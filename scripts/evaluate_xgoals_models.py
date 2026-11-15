import duckdb
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

def main():
    con = duckdb.connect("data/warehouse/dev.duckdb", read_only=True)
    df = con.sql("SELECT * FROM gold.gold_games ORDER BY game_date, game_id").pl()

    # Filter to completed regular season matches with valid results
    df = df.filter(pl.col("is_played") == True).filter(pl.col("home_win").is_not_null())
    
    # Also fetch team possession/xG proxies (Corsi %, Fenwick %, PDO) from silver_team_metrics
    tm = con.sql("SELECT team_abbrev, corsi_pct, fenwick_pct, pdo_5v5 FROM silver.silver_team_metrics WHERE season = 20242025").pl()
    
    # Join current team possession proxies to see their team-level discriminative power
    df = df.join(
        tm.rename({"team_abbrev": "home_team_abbrev", "corsi_pct": "home_team_corsi", "fenwick_pct": "home_team_fenwick", "pdo_5v5": "home_team_pdo"}),
        on="home_team_abbrev",
        how="left"
    ).join(
        tm.rename({"team_abbrev": "away_team_abbrev", "corsi_pct": "away_team_corsi", "fenwick_pct": "away_team_fenwick", "pdo_5v5": "away_team_pdo"}),
        on="away_team_abbrev",
        how="left"
    ).with_columns([
        (pl.col("home_team_corsi") - pl.col("away_team_corsi")).alias("team_corsi_diff"),
        (pl.col("home_team_fenwick") - pl.col("away_team_fenwick")).alias("team_fenwick_diff"),
        (pl.col("home_team_pdo") - pl.col("away_team_pdo")).alias("team_pdo_diff"),
    ])

    print("=" * 80)
    print(f"EVALUATION DATASET: {len(df)} games from season {df['season'].unique().to_list()}")
    print("=" * 80)

    # 1. Bivariate Correlations
    test_features = [
        "team_fenwick_diff",
        "team_corsi_diff",
        "team_pdo_diff",
        "in_season_goal_diff_per_game_diff",
        "in_season_win_pct_diff",
        "in_season_goals_for_diff_per_game",
        "in_season_goals_against_diff_per_game",
        "goals_for_diff_10",
        "goals_against_diff_10",
        "goals_for_diff_5",
        "goals_against_diff_5",
        "win_pct_diff_10",
        "win_pct_diff_5",
        "rest_days_diff",
    ]

    print("\n--- 1. BIVARIATE CORRELATIONS WITH TARGET (home_win) AND REALIZED GOAL DIFFERENTIAL (goal_diff) ---")
    corr_records = []
    for col in test_features:
        if col in df.columns:
            sub = df.select([col, "home_win", "goal_diff"]).drop_nulls()
            if len(sub) > 0:
                r_win = np.corrcoef(sub[col].to_numpy(), sub["home_win"].to_numpy())[0, 1]
                r_diff = np.corrcoef(sub[col].to_numpy(), sub["goal_diff"].to_numpy())[0, 1]
                corr_records.append({
                    "Feature": col,
                    "Corr (home_win)": round(float(r_win), 4),
                    "Corr (goal_diff)": round(float(r_diff), 4),
                    "Valid Games": len(sub),
                })
    
    corr_df = pl.DataFrame(corr_records).sort("Corr (home_win)", descending=True)
    print(corr_df)

    # 2. Chronological Split (first 75% train, last 25% test - holdout)
    split_idx = int(len(df) * 0.75)
    train_df = df.slice(0, split_idx)
    test_df = df.slice(split_idx)

    print(f"\n--- 2. CHRONOLOGICAL TRAIN / TEST SPLIT ---")
    print(f"Train games: {len(train_df)} ({train_df['game_date'].min()} to {train_df['game_date'].max()}) | Home win %: {train_df['home_win'].mean()*100:.1f}%")
    print(f"Test games:  {len(test_df)} ({test_df['game_date'].min()} to {test_df['game_date'].max()}) | Home win %: {test_df['home_win'].mean()*100:.1f}%")

    y_train = train_df["home_win"].to_numpy()
    y_test = test_df["home_win"].to_numpy()

    def get_feature_matrix(feats, tr_df, te_df):
        tr_m = []
        te_m = []
        for f in feats:
            med = float(tr_df[f].drop_nulls().median()) if tr_df[f].drop_nulls().len() > 0 else 0.0
            tr_m.append(tr_df[f].fill_null(med).to_numpy())
            te_m.append(te_df[f].fill_null(med).to_numpy())
        return np.column_stack(tr_m), np.column_stack(te_m)

    results = []

    # Model 0: Naive Prior Baseline
    prior_prob = float(y_train.mean())
    naive_probs = np.full_like(y_test, fill_value=prior_prob, dtype=float)
    naive_preds = (naive_probs >= 0.5).astype(int)
    results.append({
        "Model": "0. Naive Prior (Constant Home Win Rate)",
        "Log Loss": round(log_loss(y_test, naive_probs), 4),
        "Brier Score": round(brier_score_loss(y_test, naive_probs), 4),
        "ROC AUC": round(roc_auc_score(y_test, naive_probs), 4),
        "Accuracy (%)": round(accuracy_score(y_test, naive_preds) * 100, 2),
    })

    # Model 1: Team Identity Baseline (Joni's categorical baseline)
    team_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr_teams = team_enc.fit_transform(train_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())
    X_te_teams = team_enc.transform(test_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())
    
    m_teams = LogisticRegression(C=0.1, solver="liblinear", random_state=42)
    m_teams.fit(X_tr_teams, y_train)
    probs_teams = m_teams.predict_proba(X_te_teams)[:, 1]
    preds_teams = (probs_teams >= 0.5).astype(int)
    results.append({
        "Model": "1. Team Identity Baseline (Joni's Model)",
        "Log Loss": round(log_loss(y_test, probs_teams), 4),
        "Brier Score": round(brier_score_loss(y_test, probs_teams), 4),
        "ROC AUC": round(roc_auc_score(y_test, probs_teams), 4),
        "Accuracy (%)": round(accuracy_score(y_test, preds_teams) * 100, 2),
    })

    # Model 2: xG Proxies (Fenwick %, Corsi %, PDO)
    xg_feats = ["team_fenwick_diff", "team_corsi_diff", "team_pdo_diff"]
    X_tr_xg, X_te_xg = get_feature_matrix(xg_feats, train_df, test_df)
    scaler_xg = StandardScaler()
    X_tr_xg_s = scaler_xg.fit_transform(X_tr_xg)
    X_te_xg_s = scaler_xg.transform(X_te_xg)

    m_xg = LogisticRegression(C=1.0, random_state=42)
    m_xg.fit(X_tr_xg_s, y_train)
    probs_xg = m_xg.predict_proba(X_te_xg_s)[:, 1]
    preds_xg = (probs_xg >= 0.5).astype(int)
    results.append({
        "Model": "2. Expected Goals Proxies (Fenwick, Corsi, PDO)",
        "Log Loss": round(log_loss(y_test, probs_xg), 4),
        "Brier Score": round(brier_score_loss(y_test, probs_xg), 4),
        "ROC AUC": round(roc_auc_score(y_test, probs_xg), 4),
        "Accuracy (%)": round(accuracy_score(y_test, preds_xg) * 100, 2),
    })

    # Model 3: In-Season Expanding & Rolling Goal Metrics (Leak-free)
    goal_feats = [
        "in_season_goals_for_diff_per_game",
        "in_season_goals_against_diff_per_game",
        "in_season_goal_diff_per_game_diff",
        "goals_for_diff_5",
        "goals_against_diff_5",
        "goals_for_diff_10",
        "goals_against_diff_10",
    ]
    X_tr_g, X_te_g = get_feature_matrix(goal_feats, train_df, test_df)
    scaler_g = StandardScaler()
    X_tr_g_s = scaler_g.fit_transform(X_tr_g)
    X_te_g_s = scaler_g.transform(X_te_g)

    m_g = LogisticRegression(C=0.1, random_state=42)
    m_g.fit(X_tr_g_s, y_train)
    probs_g = m_g.predict_proba(X_te_g_s)[:, 1]
    preds_g = (probs_g >= 0.5).astype(int)
    results.append({
        "Model": "3. In-Season Goals & Momentum (Rolling 5/10 + Expanding)",
        "Log Loss": round(log_loss(y_test, probs_g), 4),
        "Brier Score": round(brier_score_loss(y_test, probs_g), 4),
        "ROC AUC": round(roc_auc_score(y_test, probs_g), 4),
        "Accuracy (%)": round(accuracy_score(y_test, preds_g) * 100, 2),
    })

    # Model 4: Combined Model (Team Identity + xG Proxies + Goal Features + Rest)
    comb_num_feats = xg_feats + goal_feats + ["rest_days_diff", "is_b2b_diff", "games_last_7_days_diff"]
    X_tr_comb_num, X_te_comb_num = get_feature_matrix(comb_num_feats, train_df, test_df)
    scaler_comb = StandardScaler()
    X_tr_comb_num_s = scaler_comb.fit_transform(X_tr_comb_num)
    X_te_comb_num_s = scaler_comb.transform(X_te_comb_num)

    X_tr_full = np.hstack([X_tr_teams, X_tr_comb_num_s])
    X_te_full = np.hstack([X_te_teams, X_te_comb_num_s])

    m_full = LogisticRegression(C=0.05, solver="liblinear", random_state=42)
    m_full.fit(X_tr_full, y_train)
    probs_full = m_full.predict_proba(X_te_full)[:, 1]
    preds_full = (probs_full >= 0.5).astype(int)
    results.append({
        "Model": "4. Full Model (Team Identity + xG Proxies + Goals + Rest)",
        "Log Loss": round(log_loss(y_test, probs_full), 4),
        "Brier Score": round(brier_score_loss(y_test, probs_full), 4),
        "ROC AUC": round(roc_auc_score(y_test, probs_full), 4),
        "Accuracy (%)": round(accuracy_score(y_test, preds_full) * 100, 2),
    })

    print("\n--- 3. OUT-OF-SAMPLE TEST SET PERFORMANCE COMPARISON ---")
    res_df = pl.DataFrame(results)
    print(res_df)

    # 4. Feature Weights in Models
    print("\n--- 4. FEATURE WEIGHTS IN xG PROXIES MODEL (Model 2) ---")
    for f, coef in zip(xg_feats, m_xg.coef_[0]):
        print(f"  {f:25s}: {coef:+.4f} (Odds ratio: {np.exp(coef):.3f})")

    print("\n--- 5. FEATURE WEIGHTS IN IN-SEASON GOALS MODEL (Model 3) ---")
    for f, coef in zip(goal_feats, m_g.coef_[0]):
        print(f"  {f:37s}: {coef:+.4f} (Odds ratio: {np.exp(coef):.3f})")

    # 5. Goal Differential Prediction (Linear Regression on goal_diff)
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr_xg_s, train_df["goal_diff"].to_numpy())
    r2_xg = ridge.score(X_te_xg_s, test_df["goal_diff"].to_numpy())
    print("\n--- 6. GOAL DIFFERENTIAL PREDICTION (RIDGE REGRESSION) ---")
    print(f"  xG Proxies (Fenwick, Corsi, PDO) R^2 on goal_diff: {r2_xg:.4f}")
    for f, coef in zip(xg_feats, ridge.coef_):
        print(f"  {f:25s}: {coef:+.4f}")

if __name__ == "__main__":
    main()
