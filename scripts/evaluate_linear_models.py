"""
Evaluates Linear and Probabilistic Models for NHL Match Outcome Prediction:
- Naive Prior Baseline (Constant Home Ice Advantage)
- Standings / Point Pct Baseline (Sarjataulukkobaseline)
- Joni's Sprint 1 Baseline (Team Identities + Home Ice with Logistic Regression)
- Regularized Logistic Regression (L2 / Ridge)
- Regularized Logistic Regression (L1 / Lasso)
- Regularized Logistic Regression (ElasticNet)
- Calibrated Ridge Classifier (Platt scaling / Sigmoid)
- Calibrated Ridge Classifier (Teams Only)

Temporal Validation Strategy:
- Training (Opetusdata): Kauden alku (tai aiemmat kaudet) - 30.11.2024
- Validation (Validointidata): 1.12.2024 - 28.2.2025
- Test (Testidata / Holdout): 1.3.2025 - runkosarjan loppuun

Includes Franchise Relocation Normalization:
- Maps Arizona Coyotes ('ARI') -> Utah Hockey Club ('UTA') for continuous franchise history.

Academic References & Theoretical Foundations:
1. Porela, M. (2024). Predicting NHL Match Outcomes Using a Data-Driven Model. Aalto University.
2. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities With Supervised Learning. ICML.
3. Swartz, T. B., & Arce, A. (2014). New insights involving the home team advantage.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
5. Weissbock, J., et al. (2013). Use of Twitter sentiment for NHL game prediction.
"""

from __future__ import annotations
import duckdb
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Franchise relocation mapping: Arizona Coyotes moved to Salt Lake City in 2024
# to become Utah Hockey Club, retaining all player contracts and hockey assets.
FRANCHISE_MAP = {
    "ARI": "UTA",
}

def normalize_team(team: str | None) -> str:
    if not team:
        return ""
    return FRANCHISE_MAP.get(team, team)

def compute_metrics(y_true: np.ndarray, probas: np.ndarray) -> dict[str, float]:
    preds = (probas >= 0.5).astype(int)
    eps = 1e-15
    probas_clipped = np.clip(probas, eps, 1.0 - eps)
    return {
        "Log Loss": round(float(log_loss(y_true, probas_clipped)), 4),
        "Brier Score": round(float(brier_score_loss(y_true, probas_clipped)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, probas_clipped)), 4),
        "Accuracy (%)": round(float(accuracy_score(y_true, preds) * 100), 2),
    }

def main():
    con = duckdb.connect("data/warehouse/dev.duckdb", read_only=True)
    df = con.sql("""
        SELECT *
        FROM gold.gold_games
        WHERE is_played = true AND game_type = 2
        ORDER BY game_date, game_id
    """).pl()

    # Apply franchise mapping to guarantee seamless team identity tracking
    df = df.with_columns([
        pl.col("home_team_abbrev").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team_abbrev"),
        pl.col("away_team_abbrev").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team_abbrev"),
    ])

    # Temporal split agreed across the project team
    train_df = df.filter(pl.col("game_date") <= pl.lit("2024-11-30").str.to_date())
    val_df = df.filter(
        (pl.col("game_date") >= pl.lit("2024-12-01").str.to_date()) &
        (pl.col("game_date") <= pl.lit("2025-02-28").str.to_date())
    )
    test_df = df.filter(pl.col("game_date") >= pl.lit("2025-03-01").str.to_date())

    y_train = train_df["home_win"].to_numpy()
    y_val = val_df["home_win"].to_numpy()
    y_test = test_df["home_win"].to_numpy()

    # Dynamic and contextual numeric features
    num_cols = [
        "rest_days_diff",
        "is_b2b_diff",
        "is_3_in_4_diff",
        "games_last_7_days_diff",
        "in_season_win_pct_diff",
        "in_season_goals_for_diff_per_game",
        "in_season_goals_against_diff_per_game",
        "in_season_goal_diff_per_game_diff",
        "win_pct_diff_5",
        "goals_for_diff_5",
        "goals_against_diff_5",
        "win_pct_diff_10",
        "goals_for_diff_10",
        "goals_against_diff_10",
    ]

    # Categorical Team identities (One-Hot Encoded)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr_teams = encoder.fit_transform(train_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())
    X_va_teams = encoder.transform(val_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())
    X_te_teams = encoder.transform(test_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())

    # Preprocess numerical features with median imputation and standard scaling
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(imputer.fit_transform(train_df.select(num_cols).to_numpy()))
    X_va_num = scaler.transform(imputer.transform(val_df.select(num_cols).to_numpy()))
    X_te_num = scaler.transform(imputer.transform(test_df.select(num_cols).to_numpy()))

    # Combined feature representations
    X_tr_comb = np.hstack([X_tr_num, X_tr_teams])
    X_va_comb = np.hstack([X_va_num, X_va_teams])
    X_te_comb = np.hstack([X_te_num, X_te_teams])

    # -------------------------------------------------------------
    # 0. Naive Home Ice Baseline (Historical class prior)
    # Ref: Swartz & Arce (2014); Porela (2024)
    # -------------------------------------------------------------
    prior = float(y_train.mean())
    val_probs_naive = np.full_like(y_val, fill_value=prior, dtype=float)
    test_probs_naive = np.full_like(y_test, fill_value=prior, dtype=float)

    # -------------------------------------------------------------
    # 1. Standings / In-Season Performance Baseline (Sarjataulukkobaseline)
    # Uses in-season win pct difference with simple logistic scaling
    # Ref: Porela (2024); Gu et al. (2014)
    # -------------------------------------------------------------
    lr_standings = LogisticRegression(solver="lbfgs", random_state=42)
    # Fit strictly on in-season win pct difference
    X_tr_standings = train_df.select(["in_season_win_pct_diff"]).fill_null(0.0).to_numpy()
    X_va_standings = val_df.select(["in_season_win_pct_diff"]).fill_null(0.0).to_numpy()
    X_te_standings = test_df.select(["in_season_win_pct_diff"]).fill_null(0.0).to_numpy()
    lr_standings.fit(X_tr_standings, y_train)
    val_probs_standings = lr_standings.predict_proba(X_va_standings)[:, 1]
    test_probs_standings = lr_standings.predict_proba(X_te_standings)[:, 1]

    # -------------------------------------------------------------
    # 2. Joni's Sprint 1 Baseline (Team IDs + Home Ice with Logistic Regression C=0.1)
    # -------------------------------------------------------------
    joni_baseline = LogisticRegression(C=0.1, solver="liblinear", random_state=42)
    joni_baseline.fit(X_tr_teams, y_train)
    val_probs_joni = joni_baseline.predict_proba(X_va_teams)[:, 1]
    test_probs_joni = joni_baseline.predict_proba(X_te_teams)[:, 1]

    # -------------------------------------------------------------
    # 3. Regularized Logistic Regression (L2 / Ridge Penalty)
    # Ref: Hastie et al. (2009)
    # -------------------------------------------------------------
    best_c_l2 = 0.05
    best_loss_l2 = float("inf")
    for c_cand in [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        m = LogisticRegression(C=c_cand, solver="lbfgs", max_iter=1000, random_state=42)
        m.fit(X_tr_comb, y_train)
        p = m.predict_proba(X_va_comb)[:, 1]
        loss = log_loss(y_val, p)
        if loss < best_loss_l2:
            best_loss_l2 = loss
            best_c_l2 = c_cand

    lr_l2 = LogisticRegression(C=best_c_l2, solver="lbfgs", max_iter=1000, random_state=42)
    lr_l2.fit(X_tr_comb, y_train)
    val_probs_l2 = lr_l2.predict_proba(X_va_comb)[:, 1]
    test_probs_l2 = lr_l2.predict_proba(X_te_comb)[:, 1]

    # -------------------------------------------------------------
    # 4. Regularized Logistic Regression (L1 / Lasso Penalty - Feature Selection)
    # Ref: Tibshirani (1996); Hastie et al. (2009)
    # -------------------------------------------------------------
    best_c_l1 = 0.05
    best_loss_l1 = float("inf")
    for c_cand in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        m = LogisticRegression(C=c_cand, solver="liblinear", penalty="l1", max_iter=1000, random_state=42)
        m.fit(X_tr_comb, y_train)
        p = m.predict_proba(X_va_comb)[:, 1]
        loss = log_loss(y_val, p)
        if loss < best_loss_l1:
            best_loss_l1 = loss
            best_c_l1 = c_cand

    lr_l1 = LogisticRegression(C=best_c_l1, solver="liblinear", penalty="l1", max_iter=1000, random_state=42)
    lr_l1.fit(X_tr_comb, y_train)
    val_probs_l1 = lr_l1.predict_proba(X_va_comb)[:, 1]
    test_probs_l1 = lr_l1.predict_proba(X_te_comb)[:, 1]

    # -------------------------------------------------------------
    # 5. Regularized Logistic Regression (ElasticNet)
    # Ref: Zou & Hastie (2005)
    # -------------------------------------------------------------
    lr_enet = LogisticRegression(C=0.05, solver="saga", penalty="elasticnet", l1_ratio=0.5, max_iter=2000, random_state=42)
    lr_enet.fit(X_tr_comb, y_train)
    val_probs_enet = lr_enet.predict_proba(X_va_comb)[:, 1]
    test_probs_enet = lr_enet.predict_proba(X_te_comb)[:, 1]

    # -------------------------------------------------------------
    # 6. Calibrated Ridge Classifier (alpha tuned on validation + Sigmoid Platt scaling)
    # Ref: Niculescu-Mizil & Caruana (2005)
    # -------------------------------------------------------------
    best_alpha_ridge = 50.0
    best_loss_ridge = float("inf")
    for alpha_cand in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]:
        base = RidgeClassifier(alpha=alpha_cand, random_state=42)
        cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        cal.fit(X_tr_comb, y_train)
        p = cal.predict_proba(X_va_comb)[:, 1]
        loss = log_loss(y_val, p)
        if loss < best_loss_ridge:
            best_loss_ridge = loss
            best_alpha_ridge = alpha_cand

    ridge_cal = CalibratedClassifierCV(estimator=RidgeClassifier(alpha=best_alpha_ridge, random_state=42), method="sigmoid", cv=3)
    ridge_cal.fit(X_tr_comb, y_train)
    val_probs_ridge = ridge_cal.predict_proba(X_va_comb)[:, 1]
    test_probs_ridge = ridge_cal.predict_proba(X_te_comb)[:, 1]

    # -------------------------------------------------------------
    # 7. Calibrated Ridge Classifier (Teams Only - Direct counterpart to Joni's LR)
    # -------------------------------------------------------------
    ridge_teams_cal = CalibratedClassifierCV(estimator=RidgeClassifier(alpha=50.0, random_state=42), method="sigmoid", cv=3)
    ridge_teams_cal.fit(X_tr_teams, y_train)
    val_probs_ridge_teams = ridge_teams_cal.predict_proba(X_va_teams)[:, 1]
    test_probs_ridge_teams = ridge_teams_cal.predict_proba(X_te_teams)[:, 1]

    # Models Comparison Table
    models_dict = {
        "0. Naive Prior Baseline (Home Ice ~54%)": (val_probs_naive, test_probs_naive),
        "1. Standings Baseline (In-Season Win % Diff)": (val_probs_standings, test_probs_standings),
        "2. Joni Sprint 1 Baseline (Teams + Home Ice LR)": (val_probs_joni, test_probs_joni),
        f"3. Regularized Logistic Reg (L2, C={best_c_l2})": (val_probs_l2, test_probs_l2),
        f"4. Regularized Logistic Reg (L1, C={best_c_l1})": (val_probs_l1, test_probs_l1),
        "5. Regularized Logistic Reg (ElasticNet)": (val_probs_enet, test_probs_enet),
        f"6. Calibrated Ridge Classifier (Combined, a={best_alpha_ridge})": (val_probs_ridge, test_probs_ridge),
        "7. Calibrated Ridge Classifier (Teams Only, a=50)": (val_probs_ridge_teams, test_probs_ridge_teams),
    }

    val_records = []
    test_records = []
    for name, (vp, tp) in models_dict.items():
        val_records.append({"Model": name, **compute_metrics(y_val, vp)})
        test_records.append({"Model": name, **compute_metrics(y_test, tp)})

    val_table = pl.DataFrame(val_records)
    test_table = pl.DataFrame(test_records)

    print("\n" + "=" * 90)
    print(f"VALIDATION SET RESULTS (1.12.2024 - 28.2.2025 | N = {len(val_df)} games)")
    print("=" * 90)
    print(val_table)

    print("\n" + "=" * 90)
    print(f"TEST SET RESULTS (1.3.2025 - runkosarjan loppu | N = {len(test_df)} games)")
    print("=" * 90)
    print(test_table)

if __name__ == "__main__":
    main()
