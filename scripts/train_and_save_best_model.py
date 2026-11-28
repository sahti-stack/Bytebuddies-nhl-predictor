"""
Train and Save Best Production Models for Bytebuddies NHL Predictor.
Authored for Product Owner evaluation and deployment.

Saves:
1. Calibrated Ridge Classifier (Platt Sigmoid, Best Probabilistic Calibrator) -> artifacts/calibrated_ridge_model.joblib
2. XGBoost Classifier (Best Tree-Based Ensemble) -> artifacts/xgboost_model.joblib
3. Default Production Model -> artifacts/best_model.joblib
4. Model Metadata & Metrics -> artifacts/model_metadata.json
"""

from __future__ import annotations
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import duckdb
import joblib
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

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
        "log_loss": round(float(log_loss(y_true, probas_clipped)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probas_clipped)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probas_clipped)), 4),
        "accuracy_pct": round(float(accuracy_score(y_true, preds) * 100), 2),
    }

def main():
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect("data/warehouse/dev.duckdb", read_only=True)
    df = con.sql("""
        SELECT *
        FROM gold.gold_games
        WHERE is_played = true AND game_type = 2
        ORDER BY game_date, game_id
    """).pl()

    # Normalize franchise relocation (Arizona -> Utah)
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

    y_train = train_df["home_win"].to_numpy().astype(int)
    y_val = val_df["home_win"].to_numpy().astype(int)
    y_test = test_df["home_win"].to_numpy().astype(int)

    # Full set of leak-free predictive features:
    # 1. Prior-season baseline strength (Corsi, Point %, Special Teams)
    # 2. Cumulative in-season performance (Win %, Goals for/against)
    # 3. Schedule fatigue and rest differentials (7-day density, B2B, 3-in-4)
    # 4. Short-term rolling form (5 and 10 games)
    num_cols = [
        # Schedule fatigue & rest
        "rest_days_diff",
        "is_b2b_diff",
        "is_3_in_4_diff",
        "games_last_7_days_diff",
        # Prior-season historical baseline (leak-free)
        "prior_point_pct_diff",
        "prior_corsi_diff",
        "prior_special_teams_diff",
        "prior_shots_diff_per_game",
        "prior_pdo_diff",
        # In-season cumulative performance
        "in_season_win_pct_diff",
        "in_season_goals_for_diff_per_game",
        "in_season_goals_against_diff_per_game",
        "in_season_goal_diff_per_game_diff",
        # Rolling recent form
        "win_pct_diff_5",
        "goals_for_diff_5",
        "goals_against_diff_5",
        "win_pct_diff_10",
        "goals_for_diff_10",
        "goals_against_diff_10",
    ]

    # Team identifiers (One-Hot Encoded)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr_teams = encoder.fit_transform(train_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())
    X_va_teams = encoder.transform(val_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())
    X_te_teams = encoder.transform(test_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())

    # Preprocess numerical features with median imputation and scaling
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(imputer.fit_transform(train_df.select(num_cols).to_numpy()))
    X_va_num = scaler.transform(imputer.transform(val_df.select(num_cols).to_numpy()))
    X_te_num = scaler.transform(imputer.transform(test_df.select(num_cols).to_numpy()))

    # Combined representations
    X_tr_comb = np.hstack([X_tr_num, X_tr_teams])
    X_va_comb = np.hstack([X_va_num, X_va_teams])
    X_te_comb = np.hstack([X_te_num, X_te_teams])

    # 1. Train Winning Calibrated Ridge Classifier (Platt Scaling)
    print("Training Calibrated Ridge Classifier (alpha=50, Sigmoid calibration)...")
    base_ridge = RidgeClassifier(alpha=50.0, random_state=42)
    cal_ridge = CalibratedClassifierCV(estimator=base_ridge, method="sigmoid", cv=3)
    cal_ridge.fit(X_tr_comb, y_train)

    val_probs_ridge = cal_ridge.predict_proba(X_va_comb)[:, 1]
    test_probs_ridge = cal_ridge.predict_proba(X_te_comb)[:, 1]
    metrics_ridge_val = compute_metrics(y_val, val_probs_ridge)
    metrics_ridge_test = compute_metrics(y_test, test_probs_ridge)

    # 2. Train Winning XGBoost Classifier
    print("Training XGBoost Classifier (max_depth=3, n_estimators=100, lr=0.05)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_tr_comb, y_train)

    val_probs_xgb = xgb_model.predict_proba(X_va_comb)[:, 1]
    test_probs_xgb = xgb_model.predict_proba(X_te_comb)[:, 1]
    metrics_xgb_val = compute_metrics(y_val, val_probs_xgb)
    metrics_xgb_test = compute_metrics(y_test, test_probs_xgb)

    # Save artifacts
    pipeline_artifacts = {
        "calibrated_ridge": cal_ridge,
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
        "num_cols": num_cols,
    }
    joblib.dump(pipeline_artifacts, artifacts_dir / "calibrated_ridge_pipeline.joblib")
    joblib.dump(xgb_model, artifacts_dir / "xgboost_model.joblib")
    joblib.dump(pipeline_artifacts, artifacts_dir / "best_model.joblib")

    metadata = {
        "selected_best_model": "Calibrated Ridge Classifier (alpha=50 + Platt Sigmoid)",
        "selected_best_tree_model": "XGBoost Classifier (max_depth=3, n_estimators=100)",
        "samples": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "calibrated_ridge_metrics": {
            "validation": metrics_ridge_val,
            "test": metrics_ridge_test,
        },
        "xgboost_metrics": {
            "validation": metrics_xgb_val,
            "test": metrics_xgb_test,
        },
        "feature_columns": num_cols,
    }

    with open(artifacts_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("SUCCESS: Best models and metadata successfully saved to artifacts/")
    print("=" * 80)
    print(f"1. VALIDATION SET (1.12.2024 - 28.2.2025 | N = {len(val_df)}):")
    print(f"   Calibrated Ridge -> Log Loss: {metrics_ridge_val['log_loss']} | Brier: {metrics_ridge_val['brier_score']} | ROC-AUC: {metrics_ridge_val['roc_auc']} | Accuracy: {metrics_ridge_val['accuracy_pct']}%")
    print(f"   XGBoost          -> Log Loss: {metrics_xgb_val['log_loss']} | Brier: {metrics_xgb_val['brier_score']} | ROC-AUC: {metrics_xgb_val['roc_auc']} | Accuracy: {metrics_xgb_val['accuracy_pct']}%")
    print("-" * 80)
    print(f"2. TEST SET / HOLDOUT (1.3.2025+ | N = {len(test_df)}):")
    print(f"   Calibrated Ridge -> Log Loss: {metrics_ridge_test['log_loss']} | Brier: {metrics_ridge_test['brier_score']} | ROC-AUC: {metrics_ridge_test['roc_auc']} | Accuracy: {metrics_ridge_test['accuracy_pct']}%")
    print(f"   XGBoost          -> Log Loss: {metrics_xgb_test['log_loss']} | Brier: {metrics_xgb_test['brier_score']} | ROC-AUC: {metrics_xgb_test['roc_auc']} | Accuracy: {metrics_xgb_test['accuracy_pct']}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
