import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import polars as pl
    import requests

    return mo, pl, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NHL API Exploration & Analysis (Mitro)

    Tämä notebook suorittaa NHL API -rajapintojen analyysin opinnäytetyö-/koneoppimisprojektia varten ja vastaa määriteltyihin tehtäviin:
    1. **API-dokumentaation analyysi** (aktiiviset vs. vanhentuneet rajapinnat)
    2. **Autentikointi ja yhteysasetukset**
    3. **Testipyyntöjen tekeminen (Kausi 2026-2027)**
    4. **Palautuvan datan rakenne ja muoto**
    5. **Rajapinnan rajoitteet ja puuttuvat kentät**
    6. **Saatavilla olevan datan kartoitus koneoppimismallia varten** (ml. Corsi/Fenwick ja NHL Edge)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. API-dokumentaation analyysi

    Projektille annettiin kolme lähdettä:
    * **Zmalski/NHL-API-Reference**: Uusien virallisten taustarajapintojen (`api-web.nhle.com` ja `api.nhle.com/stats/rest`) kattava ja ajantasainen dokumentaatio.
    * **nhl-api-py (PyPI)**: Aktiivinen Python-kääre (v3.7+), joka abstrahoi uudet endpointit selkeiksi moduuleiksi (`schedule`, `teams`, `stats`, `edge`, `game_center`).
    * **dword4/nhlapi (GitLab)**: Dokumentoi vanhan `statsapi.web.nhl.com`-rajapinnan, jonka **NHL sulki kokonaan loka-marraskuussa 2023**. Osoite ei enää vastaa (`Could not resolve host`).

    > **Yhteenveto:** Kaikki uudet pyynnöt tulee osoittaa joko `https://api-web.nhle.com/v1/` tai `https://api.nhle.com/stats/rest/en/` -palvelimille.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Autentikointi ja yhteysasetukset

    * **Autentikointi:** API on täysin julkinen eikä vaadi API-avainta, OAuth-tokenia tai rekisteröitymistä.
    * **Metodi:** Ainoastaan HTTP `GET` -pyyntöjä.
    * **Cloudflare & Headers:** Rajapinta toimii Cloudflare CDN:n takana. Käytetään `requests.Session()`-istuntoa ja asiallista `User-Agent`-otsaketta.
    """)
    return


@app.cell
def _(requests):
    # Luodaan uudelleenkäytettävä istunto asianmukaisilla otsakkeilla
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Bytebuddies-NHL-Predictor/1.0 (Student ML Project; KAMK)"
    })
    CURRENT_SEASON = "20262027"
    return CURRENT_SEASON, session


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Testipyyntöjen tekeminen (Kausi 2026-2027)

    Testataan kahta keskeistä päätepistettä kauden 2026-2027 tuoreella datalla:
    1. **Web API**: Sarjataulukko (`/v1/standings/now`)
    2. **Stats REST API**: Joukkuetason tilastokooste (`/en/team/summary?cayenneExp=seasonId=20262027...`)
    """)
    return


@app.cell
def _(CURRENT_SEASON, session):
    # Testi 1: Sarjataulukko Web API:sta (ohjautuu tuoreimpaan päivämäärään)
    standings_res = session.get("https://api-web.nhle.com/v1/standings/now")
    print(f"Standings status: {standings_res.status_code}")
    print(f"Standings URL: {standings_res.url}")
    standings_json = standings_res.json()

    # Testi 2: Joukkuekooste Stats REST API:sta (Kausi 2026-2027 runkosarja)
    stats_res = session.get(f"https://api.nhle.com/stats/rest/en/team/summary?cayenneExp=seasonId={CURRENT_SEASON}%20and%20gameTypeId=2")
    print(f"Stats REST status: {stats_res.status_code}")
    stats_json = stats_res.json()
    return standings_json, stats_json


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Palautuvan datan muoto ja käsittely

    * Data palautetaan aina **JSON**-muodossa.
    * **Lokalisaatio-objektit:** Monet nimet ovat sanakirjoja (esim. `teamName: {"default": "Carolina Hurricanes", "fr": "..."}`).
    * Polarsilla nämä saadaan helposti purettua sarakkeiksi `struct.field("default")` -kutsulla.
    """)
    return


@app.cell
def _(pl, standings_json):
    # Puretaan sarjataulukko siistiksi Polars DataFrameksi
    raw_standings = pl.DataFrame(standings_json.get("standings", []))

    df_standings = raw_standings.with_columns(
        team_name=pl.col("teamName").struct.field("default"),
        team_abbrev=pl.col("teamAbbrev").struct.field("default"),
        team_common_name=pl.col("teamCommonName").struct.field("default")
    ).select([
        "team_name",
        "team_abbrev",
        "conferenceName",
        "divisionName",
        "gamesPlayed",
        "wins",
        "losses",
        "otLosses",
        "points",
        "pointPctg",
        "goalFor",
        "goalAgainst",
        "goalDifferential",
        "l10Wins",
        "l10Losses",
        "streakCode",
    ]).sort("points", descending=True)

    df_standings.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Rajapinnan rajoitteet ja puuttuvat kentät

    1. **Rate Limiting (Cloudflare):** Ei virallisia `X-RateLimit-*` -otsakkeita. Nopea sarjatulitus voi johtaa HTTP `429` tai `403 Cloudflare challenge` -virheisiin.
       * *Ratkaisu:* `time.sleep(0.1 - 0.5)` rutiineihin, tai `nhl-api-py`:n sisäänrakennettu `sleep=True`.
    2. **Kehittyneet metriikat puuttuvat suoraan perusyhteenvedoista:**
       * `team/summary` ei sisällä Corsia tai Fenwickiä suoraan, mutta ne löytyvät erillisestä `team/percentages`-raportista (`satPct`, `usatPct`)!
    3. **Edge-seurantadata rajoittuu tuoreisiin kausiin:**
       * Laukaisu- ja luistelunopeudet sekä alueajat ovat luotettavasti saatavilla vasta kaudesta **2023–2024 alkaen**.
    4. **Stats REST -sivutus:**
       * Oletuksena data rajataan pieniin eriin (usein 5–100 riviä). Kaiken datan hakemiseen tarvitaan `limit=-1`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Saatavilla oleva data ja sovellus ML-ennustemallissa

    Kartoitetaan tärkeimmät datakokonaisuudet, joita voimme hyödyntää otteluennusteiden piirremuuttujina (features):

    ### A. Joukkuetason suoritusmetriikat (Stats REST: summary & percentages)
    """)
    return


@app.cell
def _(CURRENT_SEASON, pl, session, stats_json):
    # 1. Haetaan perustilastot (voittoprosentit, maalit, erikoistilanteet)
    raw_stats = pl.DataFrame(stats_json.get("data", []))

    df_team_summary = raw_stats.select([
        "teamFullName",
        "gamesPlayed",
        "wins",
        "losses",
        "otLosses",
        "points",
        "pointPct",
        "goalsForPerGame",
        "goalsAgainstPerGame",
        "powerPlayPct",
        "penaltyKillPct",
        "shotsForPerGame",
        "shotsAgainstPerGame",
        "faceoffWinPct",
    ]).sort("points", descending=True)

    # 2. Haetaan edistyneet laukaisu- ja tasapainometriikat (Corsi=satPct, Fenwick=usatPct, PDO=shootingPlusSavePct5v5)
    pct_res = session.get(f"https://api.nhle.com/stats/rest/en/team/percentages?cayenneExp=seasonId={CURRENT_SEASON}%20and%20gameTypeId=2")
    if pct_res.status_code == 200:
        raw_pct = pl.DataFrame(pct_res.json().get("data", []))
        df_advanced_metrics = raw_pct.select([
            "teamFullName",
            "satPct",                  # Corsi % (Kaikki laukausyritykset)
            "usatPct",                 # Fenwick % (Blokkaamattomat laukausyritykset)
            "shootingPct5v5",          # 5v5 laukaisuprosentti
            "savePct5v5",              # 5v5 torjuntaprosentti
            "shootingPlusSavePct5v5",  # PDO (Onnekkuus / regressioindikaattori)
            "zoneStartPct5v5",         # Hyökkäyspään aloitusten osuus
        ]).sort("satPct", descending=True)
    else:
        df_advanced_metrics = pl.DataFrame()

    df_team_summary.head(10)
    return (df_advanced_metrics,)


@app.cell
def _(df_advanced_metrics):
    # Näytetään Corsi (satPct), Fenwick (usatPct) ja PDO -taulukko
    df_advanced_metrics.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### B. Ottelukeskus (Gamecenter) & Play-by-Play -tapahtumat

    Ottelukohtaisista tiedoista saadaan x,y-koordinaatit ja laukausten vaarallisuus.
    Tarkastellaan tuoretta kauden 2026-2027 runkosarjaottelua (esim. Boston Bruins vs Chicago Blackhawks, Game ID `2026020217`):
    """)
    return


@app.cell
def _(pl, session):
    # Esimerkkiottelu kaudelta 2026-2027: 2026020217 (BOS vs CHI)
    sample_game_id = 2026020217
    pbp_res = session.get(f"https://api-web.nhle.com/v1/gamecenter/{sample_game_id}/play-by-play")

    if pbp_res.status_code == 200:
        pbp_data = pbp_res.json()
        plays = pbp_data.get("plays", [])

        # Suodatetaan maalit ja laukaukset koordinaatteineen
        shots = [
            {
                "timeInPeriod": p.get("timeInPeriod"),
                "period": p.get("periodDescriptor", {}).get("number"),
                "type": p.get("typeDescKey"),
                "x": p.get("details", {}).get("xCoord"),
                "y": p.get("details", {}).get("yCoord"),
                "shotType": p.get("details", {}).get("shotType"),
                "shootingPlayerId": p.get("details", {}).get("shootingPlayerId"),
                "scoringPlayerId": p.get("details", {}).get("scoringPlayerId"),
            }
            for p in plays if p.get("typeDescKey") in ["shot-on-goal", "goal", "missed-shot", "blocked-shot"]
        ]
        df_shots = pl.DataFrame(shots)
    else:
        df_shots = pl.DataFrame()

    df_shots.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### C. NHL Edge -seurantadata (Luistelunopeudet ja aluehallinta-ajat)

    NHL Edge tuottaa mikrosensorien seurantadataa luistelunopeuksista ja aluehallinnasta.
    Tarkastellaan kauden 2026-2027 tuoretta tilannetta:
    1. **Nopeimmat hyökkääjät (Skater Max Speed Top 10)**
    2. **Joukkueiden hyökkäysalueen hallinta (Team Offensive Zone Time Top 10)**
    """)
    return


@app.cell
def _(pl, session):
    # 1. Pelaajakohtainen Edge-data: Luistelunopeuksien Top 10
    edge_speed_res = session.get("https://api-web.nhle.com/v1/edge/skater-speed-top-10/F/max/now")

    if edge_speed_res.status_code == 200:
        raw_edge_speed = edge_speed_res.json()
        df_edge_speed = (
            pl.DataFrame(raw_edge_speed)
            .with_columns(
                player_name=pl.col("player").struct.field("firstName").struct.field("default") + " " + pl.col("player").struct.field("lastName").struct.field("default"),
                team=pl.col("player").struct.field("team").struct.field("abbrev"),
                position=pl.col("player").struct.field("position"),
                max_speed_kmh=pl.col("maxSpeed").struct.field("metric"),
                max_speed_mph=pl.col("maxSpeed").struct.field("imperial"),
                bursts_over_22mph=pl.col("burstsOver22"),
                bursts_20_to_22mph=pl.col("bursts20To22"),
                bursts_18_to_20mph=pl.col("bursts18To20"),
            )
            .select([
                "player_name", "team", "position", "max_speed_kmh", "max_speed_mph",
                "bursts_over_22mph", "bursts_20_to_22mph", "bursts_18_to_20mph"
            ])
        )
    else:
        df_edge_speed = pl.DataFrame()

    df_edge_speed
    return


@app.cell
def _(pl, session):
    # 2. Joukkuekohtainen Edge-data: Aluehallinta-aikojen Top 10 (Offensive Zone Time)
    edge_zone_res = session.get("https://api-web.nhle.com/v1/edge/team-zone-time-top-10/all/offensive/now")

    if edge_zone_res.status_code == 200:
        raw_edge_zone = edge_zone_res.json()
        df_edge_zone = (
            pl.DataFrame(raw_edge_zone)
            .with_columns(
                team_name=pl.col("team").struct.field("placeNameWithPreposition").struct.field("default") + " " + pl.col("team").struct.field("commonName").struct.field("default"),
                team_abbrev=pl.col("team").struct.field("abbrev"),
                offensive_zone_pct=(pl.col("offensiveZoneTime") * 100).round(2),
                neutral_zone_pct=(pl.col("neutralZoneTime") * 100).round(2),
                defensive_zone_pct=(pl.col("defensiveZoneTime") * 100).round(2),
            )
            .select([
                "team_name", "team_abbrev", "offensive_zone_pct", "neutral_zone_pct", "defensive_zone_pct"
            ])
            .sort("offensive_zone_pct", descending=True)
        )
    else:
        df_edge_zone = pl.DataFrame()

    df_edge_zone
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### D. Yhteenveto ML-piirteistä (Feature Engineering)

    Projektin ennustemallia varten API:sta voidaan muodostaa seuraavat keskeiset piirteet:
    1. **Joukkueen kuntopuntari (Rolling form):** Viimeisen 5–10 pelin pistekeskiarvo, maaliero ja laukaisuerot.
    2. **Erikoistilannetehokkuus:** `powerPlayPct` ja `penaltyKillPct`.
    3. **Edistyneet hallintamittarit:** Corsi (`satPct`), Fenwick (`usatPct`) ja Onnekkuusindeksi (`shootingPlusSavePct5v5` / PDO).
    4. **NHL Edge -aluehallinta:** `offensive_zone_pct` ja `defensive_zone_pct`.
    5. **Lepo- ja matkustusetu (Schedule fatigue):** Otteluohjelmasta (`/v1/schedule/`) laskettava `back-to-back` -otteluiden esiintyminen ja lepopäivien määrä.
    6. **Koti-/vieras-jakauma:** Koti- ja vieraspelien erilliset tilastot standings-datasta.
    """)
    return


@app.cell
def _():
    import duckdb
    import polars as pl

    con = duckdb.connect("data/warehouse/dev.duckdb", read_only=True)

    # Haetaan data suoraan Polars DataFrameksi
    df_games = con.sql("SELECT * FROM bronze.raw_games").pl()
    df_stats = con.sql("SELECT * FROM bronze.raw_team_stats").pl()

    df_games
    return (pl,)


if __name__ == "__main__":
    app.run()
