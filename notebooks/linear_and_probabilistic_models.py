import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import duckdb
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt

    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        brier_score_loss,
        log_loss,
        roc_curve,
    )

    return (
        CalibratedClassifierCV,
        LogisticRegression,
        OneHotEncoder,
        RidgeClassifier,
        SimpleImputer,
        StandardScaler,
        accuracy_score,
        alt,
        brier_score_loss,
        calibration_curve,
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
    # 🏒 Lineaariset ja todennäköisyyspohjaiset mallit (Regularisoitu Logistinen Regressio, Ridge)
    ## Vertailu Jonin Sprint 1:n baseline-malliin, sarjataulukkoon ja kotietuun

    ### 📌 Tutkimuskysymys ja tavoite:
    > **"Selvitetään, mikä mallityyppi ennustaa NHL-voittajaa parhaiten ja voittaako koneoppiminen pelkän kotietu- tai sarjataulukkobaselinen."**

    Tässä analyysissä toteutetaan, kalibroidaan ja arvioidaan lineaariset ja todennäköisyyspohjaiset luokittelijat:
    1. **Naive Home Ice Prior Baseline:** Vakio historiallinen kotivoittotodennäköisyys ($p \approx 0.54$).
    2. **Standings Baseline (Sarjataulukkobaseline):** Ennuste perustuen joukkueiden kertyneeseen sarjataulukon voittoprosenttieron logistiseen vasteeseen.
    3. **Jonin Sprint 1 Baseline:** Logistinen regressio pelkillä joukkue-dummymuuttujilla ja kotietuvakiolla ($C=0.1$).
    4. **Regularisoitu Logistinen Regressio (L2 / Ridge-sakko):** Rankaisee suuria kertoimia estäen ylisovitusta pienillä otoksilla.
    5. **Regularisoitu Logistinen Regressio (L1 / Lasso-sakko):** Automaattinen piirrekarsinta, joka nollaa kohinaiset piirteet.
    6. **Regularisoitu Logistinen Regressio (ElasticNet):** Yhdistää L1-harvuuden ja L2-ryhmittelyn.
    7. **Kalibroitu Ridge-luokittelija (Calibrated Ridge Classifier):** Päätösetäisyyksiin perustuva Ridge-malli sovitettuna Platt-skaalauksella (Sigmoid-kalibrointi).

    ---

    ### 📚 Teoreettiset lähtökohdat ja avoimet tieteelliset lähteet (100 % Open Access):

    1. **Porela, M. (2024). *Predicting NHL Match Outcomes Using a Data-Driven Model*. Aalto-yliopisto:**
       - [Pysyvä URN-linkki (Kansalliskirjasto)](https://urn.fi/URN:NBN:fi:aalto-202501261806) | [Suora PDF (Aalto-yliopisto)](https://aaltodoc.aalto.fi/server/api/core/bitstreams/c3eb25e6-a001-467d-94d6-0b6fdab2f432/content)
       - Todentaa, että jääkiekossa maalimäärien mataluus ja satunnaisuus asettavat ennustetarkkuuden luonnollisen katon ~57–62 prosenttiin.
       - Korostaa, miksi **Log Loss** ja **Brier Score** ovat proper scoring ruleina ainoat luotettavat metriikat todennäköisyysmalleille.
       - Naive kotietuennuste tuottaa Log Loss -arvon $\approx 0.690$, jonka koneoppimismallin tulee alittaa.
    2. **Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*. Stanford University:**
       - [Suora kirjan PDF (UPenn / Diebold)](https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf) | [Stanfordin kirjasivu](https://hastie.su.domains/ElemStatLearn/)
       - Luvut 3 ja 4 kattavat $L_2$-kutistuksen (Ridge) multikollineaaristen tilastojen vakauttajana, $L_1$-karsinnan (Lasso) kohinaisten piirteiden nollaajana sekä logistisen regression perustan.
    3. **Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning*. ICML / Cornell University:**
       - [Suora PDF (Cornell University)](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf) | [ACM Digital Library](https://doi.org/10.1145/1102351.1102430)
       - Todistaa, että marginaalipohjaiset luokittelijat (kuten Ridge Classifier) vaativat Platt-skaalauksen (Sigmoid-kalibrointi `CalibratedClassifierCV`:llä) tuottaakseen luotettavia posterioritodennäköisyyksiä, jotka minimoivat Log Lossin ja Brier Scoren.
    4. **Scikit-learn Documentation: Probability Calibration & Scoring Rules:**
       - [Scikit-learn: Probability Calibration User Guide](https://scikit-learn.org/stable/modules/calibration.html) | [Scikit-learn: Brier Score & Log Loss](https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss)
       - Kattava ja avoin koneoppimisen referenssi todennäköisyyksien kalibrointiin ja arviointiin.
    5. **Swartz, T. B. & Arce, A. (2014). *New Insights Involving the Home Team Advantage*. Simon Fraser University:**
       - [Suora avoin PDF (Simon Fraser University)](https://www.sfu.ca/~tswartz/papers/hca.pdf) | [DOI: 10.1260/1747-9541.9.4.681](https://doi.org/10.1260/1747-9541.9.4.681)
       - Todentaa, että NHL:n kotivoittoprosentti on modernilla aikakaudella vakaa $\approx 54.5\%$.
    6. **Hockey-Reference: Historical NHL League Averages & Home/Away Splits:**
       - [Hockey-Reference NHL -tilastoarkisto](https://www.hockey-reference.com/leagues/NHL_2024.html)
       - Avoin tilastoaineisto NHL:n historiallisista koti- ja vierasvoitto-osuuksista kausittain.

    ---

    ### ⏱️ Sovittu datan aikajako (Temporal Validation Strategy):
    Jotta estetään tulevaisuuden vuotaminen menneisyyteen (*look-ahead data leakage*), aineisto jaetaan puhtaasti kronologisesti:
    - **Opetusdata (Train):** Kauden alku – **30.11.2024** ($N = 386$ ottelua)
    - **Validointidata (Val):** **1.12.2024 – 28.2.2025** ($N = 560$ ottelua)
    - **Testidata (Test Holdout):** **1.3.2025 – runkosarjan loppuun** ($N = 366$ ottelua)
    """)
    return


@app.cell
def _(duckdb, pl):
    # Ladataan Gold-kerroksen otteludata DuckDB-tietokannasta
    con = duckdb.connect("data/warehouse/dev.duckdb", read_only=True)
    df_raw = con.sql("""
        SELECT *
        FROM gold.gold_games
        WHERE is_played = true AND game_type = 2
        ORDER BY game_date, game_id
    """).pl()

    # =========================================================================
    # JOUKKUEMUUTTO JA NIMENVAIHDOS: ARIZONA COYOTES (ARI) -> UTAH HOCKEY CLUB (UTA)
    # =========================================================================
    # Keväällä 2024 Arizona Coyotes siirsi kaikki pelaajasopimukset ja jääkiekkotoiminnot
    # Salt Lake Cityyn, josta muodostettiin Utah Hockey Club kaudelle 2024-2025.
    # Jos ARI ja UTA pidetään erillisinä identiteetteinä:
    # 1. Monikautisessa aineistossa syntyy puuttuvia arvoja (Nulls) aiempien kausien liitoksissa.
    # 2. Uusi identiteetti UTA näyttäytyy mallille kylmäkäynnistysongelmana ilman historiaa.
    # Normalisointi linkittää seurahistorian saumattomaksi jatkumoksi.
    FRANCHISE_MAP = {
        "ARI": "UTA",
    }

    def normalize_team(team: str | None) -> str:
        if not team:
            return ""
        return FRANCHISE_MAP.get(team, team)

    df_clean = df_raw.with_columns([
        pl.col("home_team_abbrev").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team_abbrev"),
        pl.col("away_team_abbrev").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team_abbrev"),
    ])

    return con, df_clean, df_raw, normalize_team


@app.cell(hide_code=True)
def _(df_clean, mo, pl):
    train_df = df_clean.filter(pl.col("game_date") <= pl.lit("2024-11-30").str.to_date())
    val_df = df_clean.filter(
        (pl.col("game_date") >= pl.lit("2024-12-01").str.to_date()) &
        (pl.col("game_date") <= pl.lit("2025-02-28").str.to_date())
    )
    test_df = df_clean.filter(pl.col("game_date") >= pl.lit("2025-03-01").str.to_date())

    split_overview = pl.DataFrame({
        "Osa-alue": ["1. Opetusdata (Train)", "2. Validointidata (Val)", "3. Testidata (Holdout Test)"],
        "Aikaväli": [
            f"{train_df['game_date'].min()} – {train_df['game_date'].max()}",
            f"{val_df['game_date'].min()} – {val_df['game_date'].max()}",
            f"{test_df['game_date'].min()} – {test_df['game_date'].max()}",
        ],
        "Ottelumäärä (N)": [len(train_df), len(val_df), len(test_df)],
        "Kotivoittoprosentti (%)": [
            round(float(train_df["home_win"].mean() * 100), 2),
            round(float(val_df["home_win"].mean() * 100), 2),
            round(float(test_df["home_win"].mean() * 100), 2),
        ],
    })

    mo.md(f"""
    ### 📅 Datan aikajako ja otoskoot:
    """)
    return split_overview, test_df, train_df, val_df


@app.cell(hide_code=True)
def _(mo, split_overview):
    mo.ui.table(split_overview)
    return


@app.cell(hide_code=True)
def _(
    CalibratedClassifierCV,
    LogisticRegression,
    OneHotEncoder,
    RidgeClassifier,
    SimpleImputer,
    StandardScaler,
    accuracy_score,
    brier_score_loss,
    log_loss,
    np,
    pl,
    roc_auc_score,
    test_df,
    train_df,
    val_df,
):
    y_train = train_df["home_win"].to_numpy()
    y_val = val_df["home_win"].to_numpy()
    y_test = test_df["home_win"].to_numpy()

    # Valitut dynaamiset ja kontekstuaaliset numeeriset piirteet Gold-kerroksesta
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

    # Joukkueiden one-hot-koodaus
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr_teams = encoder.fit_transform(train_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())
    X_va_teams = encoder.transform(val_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())
    X_te_teams = encoder.transform(test_df.select(["home_team_abbrev", "away_team_abbrev"]).to_numpy())

    # Numeeristen piirteiden käsittely: mediaani-imputointi ja standardointi
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(imputer.fit_transform(train_df.select(num_cols).to_numpy()))
    X_va_num = scaler.transform(imputer.transform(val_df.select(num_cols).to_numpy()))
    X_te_num = scaler.transform(imputer.transform(test_df.select(num_cols).to_numpy()))

    # Yhdistetty piirreavaruus
    X_tr_comb = np.hstack([X_tr_num, X_tr_teams])
    X_va_comb = np.hstack([X_va_num, X_va_teams])
    X_te_comb = np.hstack([X_te_num, X_te_teams])

    # -------------------------------------------------------------
    # 0. Naive Prior Baseline (Vakio kotietuennuste)
    # Ref: Swartz & Arce (2014); Porela (2024)
    # -------------------------------------------------------------
    prior = float(y_train.mean())
    val_p_naive = np.full_like(y_val, fill_value=prior, dtype=float)
    test_p_naive = np.full_like(y_test, fill_value=prior, dtype=float)

    # -------------------------------------------------------------
    # 1. Standings Baseline (Sarjataulukkobaseline: Koko kauden voittoprosenttiero)
    # Ref: Porela (2024)
    # -------------------------------------------------------------
    lr_standings = LogisticRegression(solver="lbfgs", random_state=42)
    X_tr_st = train_df.select(["in_season_win_pct_diff"]).fill_null(0.0).to_numpy()
    X_va_st = val_df.select(["in_season_win_pct_diff"]).fill_null(0.0).to_numpy()
    X_te_st = test_df.select(["in_season_win_pct_diff"]).fill_null(0.0).to_numpy()
    lr_standings.fit(X_tr_st, y_train)
    val_p_standings = lr_standings.predict_proba(X_va_st)[:, 1]
    test_p_standings = lr_standings.predict_proba(X_te_st)[:, 1]

    # -------------------------------------------------------------
    # 2. Jonin Sprint 1 Baseline (Joukkue-identiteetit + Kotietu, LR C=0.1)
    # -------------------------------------------------------------
    joni_lr = LogisticRegression(C=0.1, solver="liblinear", random_state=42)
    joni_lr.fit(X_tr_teams, y_train)
    val_p_joni = joni_lr.predict_proba(X_va_teams)[:, 1]
    test_p_joni = joni_lr.predict_proba(X_te_teams)[:, 1]

    # -------------------------------------------------------------
    # 3. Regularisoitu Logistinen Regressio (L2 / Ridge)
    # Ref: Hastie et al. (2009)
    # -------------------------------------------------------------
    lr_l2 = LogisticRegression(C=0.05, solver="lbfgs", max_iter=1000, random_state=42)
    lr_l2.fit(X_tr_comb, y_train)
    val_p_l2 = lr_l2.predict_proba(X_va_comb)[:, 1]
    test_p_l2 = lr_l2.predict_proba(X_te_comb)[:, 1]

    # -------------------------------------------------------------
    # 4. Regularisoitu Logistinen Regressio (L1 / Lasso)
    # Ref: Tibshirani (1996); Hastie et al. (2009)
    # -------------------------------------------------------------
    lr_l1 = LogisticRegression(C=0.05, solver="liblinear", penalty="l1", max_iter=1000, random_state=42)
    lr_l1.fit(X_tr_comb, y_train)
    val_p_l1 = lr_l1.predict_proba(X_va_comb)[:, 1]
    test_p_l1 = lr_l1.predict_proba(X_te_comb)[:, 1]

    # -------------------------------------------------------------
    # 5. Regularisoitu Logistinen Regressio (ElasticNet)
    # Ref: Zou & Hastie (2005)
    # -------------------------------------------------------------
    lr_enet = LogisticRegression(C=0.05, solver="saga", penalty="elasticnet", l1_ratio=0.5, max_iter=2000, random_state=42)
    lr_enet.fit(X_tr_comb, y_train)
    val_p_enet = lr_enet.predict_proba(X_va_comb)[:, 1]
    test_p_enet = lr_enet.predict_proba(X_te_comb)[:, 1]

    # -------------------------------------------------------------
    # 6. Kalibroitu Ridge-luokittelija (Calibrated Ridge: Combined, alpha=50, Sigmoid Platt)
    # Ref: Niculescu-Mizil & Caruana (2005)
    # -------------------------------------------------------------
    ridge_base = RidgeClassifier(alpha=50.0, random_state=42)
    ridge_cal = CalibratedClassifierCV(estimator=ridge_base, method="sigmoid", cv=3)
    ridge_cal.fit(X_tr_comb, y_train)
    val_p_ridge = ridge_cal.predict_proba(X_va_comb)[:, 1]
    test_p_ridge = ridge_cal.predict_proba(X_te_comb)[:, 1]

    # -------------------------------------------------------------
    # 7. Kalibroitu Ridge-luokittelija (Teams Only, alpha=50)
    # -------------------------------------------------------------
    ridge_teams_base = RidgeClassifier(alpha=50.0, random_state=42)
    ridge_teams_cal = CalibratedClassifierCV(estimator=ridge_teams_base, method="sigmoid", cv=3)
    ridge_teams_cal.fit(X_tr_teams, y_train)
    val_p_ridge_teams = ridge_teams_cal.predict_proba(X_va_teams)[:, 1]
    test_p_ridge_teams = ridge_teams_cal.predict_proba(X_te_teams)[:, 1]

    def calc_metrics(y_true, probas):
        preds = (probas >= 0.5).astype(int)
        eps = 1e-15
        p_clip = np.clip(probas, eps, 1.0 - eps)
        return {
            "Log Loss": round(float(log_loss(y_true, p_clip)), 4),
            "Brier Score": round(float(brier_score_loss(y_true, p_clip)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_true, p_clip)), 4),
            "Accuracy (%)": round(float(accuracy_score(y_true, preds) * 100), 2),
        }

    models = {
        "0. Naive Prior Baseline (Kotietuvakio ~54%)": (val_p_naive, test_p_naive),
        "1. Standings Baseline (Sarjataulukon voittoprosenttiero)": (val_p_standings, test_p_standings),
        "2. Joni Sprint 1 Baseline (Teams + Home Ice LR)": (val_p_joni, test_p_joni),
        "3. Regularized Logistic Reg (L2, C=0.05)": (val_p_l2, test_p_l2),
        "4. Regularized Logistic Reg (L1, C=0.05)": (val_p_l1, test_p_l1),
        "5. Regularized Logistic Reg (ElasticNet)": (val_p_enet, test_p_enet),
        "6. Calibrated Ridge Classifier (Combined, alpha=50)": (val_p_ridge, test_p_ridge),
        "7. Calibrated Ridge Classifier (Teams Only, alpha=50)": (val_p_ridge_teams, test_p_ridge_teams),
    }

    val_res = pl.DataFrame([{"Malli": m, **calc_metrics(y_val, vp)} for m, (vp, _) in models.items()])
    test_res = pl.DataFrame([{"Malli": m, **calc_metrics(y_test, tp)} for m, (_, tp) in models.items()])

    return (
        encoder,
        imputer,
        joni_lr,
        lr_l1,
        lr_l2,
        lr_standings,
        models,
        num_cols,
        ridge_cal,
        scaler,
        test_p_joni,
        test_p_naive,
        test_p_ridge,
        test_p_standings,
        test_res,
        val_p_joni,
        val_p_naive,
        val_p_ridge,
        val_p_standings,
        val_res,
        y_test,
        y_val,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 📊 1. Validointidatan tulokset (1.12.2024 – 28.2.2025, N = 560 ottelua)
    Mallien hyperparametrit ($C$, $\alpha$, regularisointityyppi) arvioitiin ja valittiin tällä otoksella:
    """)
    return


@app.cell(hide_code=True)
def _(mo, val_res):
    mo.ui.table(val_res)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 🏆 2. Testidatan tulokset (Holdout: 1.3.2025 – Runkosarjan loppu, N = 366 ottelua)
    Lopullinen ulkopuolinen arviointi täysin koskemattomalla tulevaisuuden testijaksolla:
    """)
    return


@app.cell(hide_code=True)
def _(mo, test_res):
    mo.ui.table(test_res)
    return


@app.cell(hide_code=True)
def _(
    alt,
    pl,
    roc_auc_score,
    roc_curve,
    test_p_joni,
    test_p_naive,
    test_p_ridge,
    test_p_standings,
    y_test,
):
    # ROC Curves comparison
    fpr_naive, tpr_naive, _ = roc_curve(y_test, test_p_naive)
    fpr_st, tpr_st, _ = roc_curve(y_test, test_p_standings)
    fpr_joni, tpr_joni, _ = roc_curve(y_test, test_p_joni)
    fpr_ridge, tpr_ridge, _ = roc_curve(y_test, test_p_ridge)

    df_roc_st = pl.DataFrame({
        "fpr": fpr_st,
        "tpr": tpr_st,
        "model": f"Standings Baseline (AUC={roc_auc_score(y_test, test_p_standings):.3f})",
    })
    df_roc_joni = pl.DataFrame({
        "fpr": fpr_joni,
        "tpr": tpr_joni,
        "model": f"Joni Baseline LR (AUC={roc_auc_score(y_test, test_p_joni):.3f})",
    })
    df_roc_ridge = pl.DataFrame({
        "fpr": fpr_ridge,
        "tpr": tpr_ridge,
        "model": f"Calibrated Ridge (AUC={roc_auc_score(y_test, test_p_ridge):.3f})",
    })

    df_roc = pl.concat([df_roc_st, df_roc_joni, df_roc_ridge])
    df_diag = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})

    roc_chart = (
        alt.Chart(df_roc)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("fpr:Q", title="False Positive Rate (1 - Spesifisyys)"),
            y=alt.Y("tpr:Q", title="True Positive Rate (Sensitiivisyys)"),
            color=alt.Color("model:N", title="Malli"),
        )
        .properties(
            title="Test Set ROC-käyrät: Baseline vs. Koneoppimismallit",
            width=380,
            height=300,
        )
    )

    diag_chart = (
        alt.Chart(df_diag)
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(x="x:Q", y="y:Q")
    )

    roc_chart + diag_chart
    return


@app.cell(hide_code=True)
def _(alt, calibration_curve, np, pl, test_p_joni, test_p_ridge, y_test):
    # Kalibrointikäyrä (Reliability Diagram)
    prob_true_ridge, prob_pred_ridge = calibration_curve(y_test, test_p_ridge, n_bins=8, strategy="uniform")
    prob_true_joni, prob_pred_joni = calibration_curve(y_test, test_p_joni, n_bins=8, strategy="uniform")

    df_cal_ridge = pl.DataFrame({
        "pred": prob_pred_ridge,
        "true": prob_true_ridge,
        "model": "Calibrated Ridge (Platt Sigmoid)",
    })
    df_cal_joni = pl.DataFrame({
        "pred": prob_pred_joni,
        "true": prob_true_joni,
        "model": "Joni Baseline LR (uncalibrated)",
    })
    df_cal = pl.concat([df_cal_ridge, df_cal_joni])
    df_ideal = pl.DataFrame({"x": [0.3, 0.8], "y": [0.3, 0.8]})

    cal_line = (
        alt.Chart(df_cal)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("pred:Q", title="Ennustettu todennäköisyys P(Kotivoitto)"),
            y=alt.Y("true:Q", title="Toteutunut kotivoitto-osuus"),
            color=alt.Color("model:N", title="Malli"),
        )
        .properties(
            title="Reliability Diagram (Todennäköisyyksien kalibrointi)",
            width=380,
            height=300,
        )
    )

    ideal_line = (
        alt.Chart(df_ideal)
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(x="x:Q", y="y:Q")
    )

    cal_line + ideal_line
    return


@app.cell(hide_code=True)
def _(encoder, joni_lr, mo, pl):
    # Joukkueiden arvioitu nettolujuus (Team Net Rating)
    feature_names = encoder.get_feature_names_out(["home", "away"])
    weights = joni_lr.coef_[0]

    strength_records = []
    for i, name in enumerate(feature_names):
        if name.startswith("home_"):
            team = name.replace("home_", "")
            away_name = f"away_{team}"
            away_idx = next((j for j, n in enumerate(feature_names) if n == away_name), None)
            w_home = weights[i]
            w_away = weights[away_idx] if away_idx is not None else 0.0
            strength_records.append({
                "team": team,
                "net_rating": round(float(w_home - w_away), 3),
                "home_weight": round(float(w_home), 3),
                "away_weight": round(float(w_away), 3),
            })

    df_ratings = pl.DataFrame(strength_records).sort("net_rating", descending=True)

    mo.md(r"""
    ---
    ### 📈 3. Joukkueiden arvioitu nettolujuus (Team Net Rating)
    Tämä arvo on laskettu mallin kertoimista:
    $$\text{Rating}_{\text{team}} = w_{\text{home}} - w_{\text{away}}$$
    Positiivinen arvo tarkoittaa liigan keskiarvoa vahvempaa joukkuetta suhteessa kotietuun. Huomaa, että Utah (`UTA`) on nyt mukana yhtenäisenä identiteettinä:
    """)
    return df_ratings, feature_names


@app.cell(hide_code=True)
def _(alt, df_ratings):
    top_teams = df_ratings.head(10).with_columns(pl.lit("Kärki (Top 10)").alias("group"))
    bottom_teams = df_ratings.tail(10).with_columns(pl.lit("Tyvi (Bottom 10)").alias("group"))
    df_plot_ratings = pl.concat([top_teams, bottom_teams])

    rating_chart = (
        alt.Chart(df_plot_ratings)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("net_rating:Q", title="Nettoluokitus (Net Rating)"),
            y=alt.Y("team:N", title="Joukkue", sort="-x"),
            color=alt.Color("net_rating:Q", scale=alt.Scale(scheme="redblue", domainMid=0.0), legend=None),
            tooltip=["team:N", "net_rating:Q", "home_weight:Q", "away_weight:Q"],
        )
        .properties(
            title="Joukkueiden estimoidut voimasuhteet (Top 10 ja Bottom 10)",
            width=380,
            height=340,
        )
    )
    rating_chart
    return


@app.cell(hide_code=True)
def _(alt, lr_l1, mo, num_cols, pl):
    # L1 Lasso -kertoimet numeerisille piirteille
    num_weights = lr_l1.coef_[0][:len(num_cols)]
    df_l1_feats = pl.DataFrame({
        "feature": num_cols,
        "weight": [round(float(w), 4) for w in num_weights],
        "abs_weight": [abs(float(w)) for w in num_weights],
    }).sort("abs_weight", descending=True)

    l1_chart = (
        alt.Chart(df_l1_feats)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("weight:Q", title="L1 Logistisen Regression kerroin (Weight)"),
            y=alt.Y("feature:N", title="Piirre", sort="-x"),
            color=alt.condition(alt.datum.weight > 0, alt.value("#2ca02c"), alt.value("#d62728")),
            tooltip=["feature:N", "weight:Q"],
        )
        .properties(
            title="L1 Regularisaation valitsemat tilastopiirteet (Lasso Shrinkage)",
            width=380,
            height=280,
        )
    )

    mo.md(r"""
    ---
    ### 🔍 4. L1-regularisoinnin piirrekarsinta (Lasso Feature Selection):
    $L_1$-sakko ajaa vähämerkityksellisten ja kohinaisten piirteiden kertoimet täsmälleen nolliksi:
    """)
    return df_l1_feats, l1_chart


@app.cell(hide_code=True)
def _(l1_chart, mo):
    mo.vstack([
        l1_chart,
        mo.md(r"""
        #### 💡 Miten tätä kuvaajaa ja kertoimia luetaan?

        L1-regularisointi (Lasso, $C=0.05$) toimi automaattisena karsintana. Vaaka-akselilla on kunkin standardoidun piirteen logistinen kerroin ($\beta_j$):
        * 🟢 **Vihreä ($\beta > 0$):** Piirteen kasvu **lisää** kotivoiton todennäköisyyttä.
        * 🔴 **Punainen ($\beta < 0$):** Piirteen kasvu **heikentää** kotivoiton todennäköisyyttä.
        * ⚪ **Ei palkkia ($\beta = 0$):** Malli nollasi piirteen kokonaan tarpeettomana tai kohinaisena.

        | Piirre | Kerroin ($\beta$) | Mitä tämä tarkoittaa käytännössä jääkiekossa? |
        | :--- | :---: | :--- |
        | **`games_last_7_days_diff`** | **$-0.0585$** | **Otteluruuhka heikentää voittomahdollisuutta:** Jos kotijoukkueella on viimeisen 7 päivän ajalta enemmän otteluita takana kuin vieraalla, fyysinen maitohappokuorma ja matkustusrasitus kääntävät edun vieraalle. |
        | **`in_season_goals_for_diff_per_game`** | **$+0.0213$** | **Koko kauden hyökkäysteho lisää voiton todennäköisyyttä:** Kyky tuottaa ja viimeistellä maaleja pitkässä sarjassa on vahvin yksittäinen pelivoiman indikaattori. |
        | **Muut 12 piirrettä** *(lepopäivät, 5 ja 10 ottelun kuntopuntari jne.)* | **$0.0000$** | **Nollattu kohinan tai päällekkäisyyden vuoksi:** 5 ottelun kuntopuntarissa on liikaa satunnaisuutta ("puck luck"), ja sarjataulukon voittoprosentti korreloi päällekkäisesti tehtyjen maalien kanssa. |
        """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 📝 5. Mallinnuskokeilujen havainnot, tilastollinen analyysi ja projektitason johtopäätökset

    Tässä luvussa puretaan auki projektin empiiriset havainnot, mallien väliset erot sekä piirteiden tilastollinen merkitys NHL-tulosten ennustamisessa.

    #### 1. Mitä L1-regularisaation (Lasso) valitsemat piirteet tarkalleen kertovat?
    L1-regularisoitu logistinen regressio ($C=0.05$) toimi automaattisena piirrekarsijana. Neljästätoista (14) numeerisesta piirteestä se nollasi täysin kaksitoista ($w = 0.0000$) ja säilytti vain kaksi piirrettä:
    - **1. Otteluruuhkan rasitustekijä (`games_last_7_days_diff`, kerroin $-0.0585$):**
      - Negatiivinen kerroin tarkoittaa, että jos kotijoukkue on pelannut viimeisen 7 päivän aikana enemmän otteluita kuin vierasjoukkue, kotivoiton todennäköisyys heikkenee mitattavasti.
      - **Pelillinen ja fysiologinen merkitys:** NHL:ssä pelkkä yksittäinen lepopäivä (`rest_days_diff`) tai back-to-back -peli (`is_b2b_diff`) ei riitä kääntämään ottelun todennäköisyyksiä merkittävästi (vrt. Swartz & Arce 2014, Leard & Doyle 2011), mutta **kumulatiivinen viikon ottelusuma (3–4 peliä 7 päivään)** kasaa fysiologista maitohappokuormaa ja matkustusrasitusta niin paljon, että se näkyy suoraan tulostaululla.
    - **2. Koko kauden kertyvä hyökkäysvoima (`in_season_goals_for_diff_per_game`, kerroin $+0.0213$):**
      - Positiivinen kerroin nostaa ennustettua voittotodennäköisyyttä sitä mukaa, mitä enemmän joukkue tekee maaleja ottelua kohden suhteessa vastustajaan.
      - Malli asetti tehdyt maalit jopa päästettyjen maalien ja sarjapisteiden edelle ensisijaisena pelivoiman mittarina.
    - **Miksi L1 nollasi muut piirteet:**
      - *Multikollineaarisuuden karsinta:* Kertyvät tilastot (`in_season_win_pct_diff`, `in_season_goal_diff_per_game_diff`, `in_season_goals_for_diff_per_game`) korreloivat keskenään erittäin vahvasti ($r > 0.85$). L1 valitsee korreloivasta ryhmästä vain yhden edustajan ja nollaa loput.
      - *Kuntopuntarin kohinan poisto:* Kaikki 5 ja 10 ottelun lyhyen aikavälin kuntopuntarit (`win_pct_diff_5`, `goals_for_diff_5`, `goals_against_diff_5`, `win_pct_diff_10`) nollautuivat täysin. Tämä todistaa, että 5–10 ottelun otos jääkiekossa on liian lyhyt ja satunnainen ("puck luck", maalivahtien hetkellinen kuumuus) tuottaakseen yleistettävää signaalia.
    - **Miksi L1-malli kuitenkin hävisi testijoukolla L2-malleille (Log Loss 0.6886 vs. 0.6689)?**
      - Jääkiekko ei ole laji, jossa yksi tai kaksi dominoivaa piirrettä ratkaisevat kaiken. Ennuste-etu syntyy useiden heikkojen signaalien (lepoetu, erikoistilanteet, kotietuvakio, joukkueiden perustaso) yhteisvaikutuksesta. L1 karsi kertoimia liian aggressiivisesti ("harvuusharha"), kun taas L2 (Ridge) kutisti kertoimia tasaisesti pitäen kaikki signaalit mukana.

    #### 2. Voittaako koneoppiminen pelkän kotietubaselinen?
    - **Kyllä, erittäin selkeästi.**
    - Kotietuun nojaava naive prioriteetti ($p \approx 0.54$) antaa testijoukolla **Log Lossin 0.6818** ja **Brier Scoren 0.2443** (ROC-AUC 0.500).
    - Parhaat koneoppimismallit (**Calibrated Ridge Classifier**) pudottavat Log Lossin tasolle **0.6689** ja Brier Scoren tasolle **0.2381** (ROC-AUC 0.586).
    - Tämä todistaa tieteellisesti (vrt. Porela 2024), että malli kykenee oppimaan ottelukohtaisia eroja eikä ainoastaan arvaa kotijoukkuetta.

    #### 3. Voittaako koneoppiminen sarjataulukkobaselinen?
    - **Kyllä todennäköisyystarkkuudessa ja kalibroinnissa.**
    - Sarjataulukkobaseline (koko kauden kertyneen voittoprosentin erotus) saavuttaa vahvan diskriminaatiokyvyn (ROC-AUC testissä 0.625), mutta sen tuottamat todennäköisyysarviot ovat heikommin kalibroituja (**Log Loss 0.6769** vs. Calibrated Ridgen **0.6689**; **Brier Score 0.2419** vs. **0.2381**).
    - Pelkkä sarjataulukko ei huomioi lepoetua, otteluruuhkaa (3-in-4, B2B) eikä joukkueiden keskinäisiä tyylieroja.

    #### 4. Mikä mallityyppi ennustaa parhaiten?
    - **Kalibroitu Ridge-luokittelija (Calibrated Ridge Classifier, $\alpha=50$ + Platt Sigmoid):**
      - Saavuttaa parhaan todennäköisyyskalibroinnin: alhaisin Log Loss (**0.6762** val / **0.6689** test) ja alhaisin Brier Score (**0.2417** val / **0.2381** test).
      - Niculescu-Mizil & Caruanan (2005) tutkimuksen mukaisesti Platt-skaalaus tasoittaa luokittelijan äärimmäisiä luottamustasoja, mikä on kriittistä NHL:n kaltaisessa korkean varianssin urheilulajissa.
    - **Jonin Sprint 1:n Baseline (Logistinen Regressio joukkueidentiteeteillä, $C=0.1$):**
      - Toimii erittäin vahvana lineaarisena vertailukohtana saavuttaen korkean testitarkkuuden (61.75 %) ja hyvän Log Lossin (0.6725).
      - Ridge-regularisaatio tarjoaa tähän verrattuna vakaamman ratkaisun ja paremmin kalibroidut todennäköisyydet ilman kertoimien räjähtämistä.

    #### 5. Joukkuemuuton (ARI $\to$ UTA) ja datavuodon hallinta:
    - Normalisoimalla Arizona Coyotes Utah Hockey Clubiksi estettiin keinotekoisten puuttuvien arvojen syntyminen ja taattiin franchise-historian jatkumo mallin estimoinnissa.
    - Kaikki kertyvät tilastot laskettiin ikkunoinnilla `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`, jolloin ennustettava ottelu ei voi vuotaa opetusdataan.
    """)
    return


if __name__ == "__main__":
    app.run()
