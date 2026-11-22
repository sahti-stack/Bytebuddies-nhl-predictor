# dbt-käyttöohje (Bytebuddies NHL Predictor)

Tämä ohje opastaa dbt Medallion -dataputken käytössä ja ylläpidossa.  
Putki lukee NHL:n raakadatan (Bronze), puhdistaa ja rikastaa sen (Silver) sekä tuottaa valmiin koneoppimisen syöteaineiston (Gold) paikalliseen DuckDB-tietokantaan.

---

## 1. Valmistelu ja ympäristö
Varmista, että olet projektin juurikansiossa ja olet alustanut Python-riippuvuudet `uv`:lla:
```bash
uv sync
```

---

## 2. Datan haku (Bronze Ingestion)
Ennen dbt-mallien ajoa DuckDB-kannassa täytyy olla raakadataa. Raakadata haetaan suoraan NHL API:sta Python-skriptillä:

```bash
# Esimerkki: Haetaan kauden 2024–2025 ottelut ja tilastot
uv run python src/bronze_ingestion.py --seasons 20242025

# Esimerkki: Useampi kausi kerralla
uv run python src/bronze_ingestion.py --seasons 20222023 20232024 20242025 20262027
```

---

## 3. dbt-projektin ajaminen
Kaikki dbt-komennot suoritetaan `dbt_project`-kansiossa:

```bash
cd dbt_project
```

### Yhteyden tarkistus
```bash
uv run dbt debug
```

### Mallien suoritus (Silver & Gold)
```bash
uv run dbt run
```
Tämä komento:
1. Lukee `bronze`-skeeman raakataulut (`raw_games`, `raw_team_stats`, `raw_team_percentages`).
2. Puhdistaa ja tyypittää tiedot `silver`-skeemaan (`stg_games`, `stg_team_stats`, `stg_team_percentages`).
3. Muodostaa joukkuejatkuvuuden ristiviittauksen `silver.silver_team_lineage` (virallinen `team_id` -> synteettinen `lineage_id`).
4. Yhdistää kausitilastot joukkueprofiiliksi `silver.silver_team_metrics` ja lisää siihen `lineage_id`-avaimen.
5. Laskee rikastukset ja ikkunafunktiot (lepopäivät, kuntopuntarit) tauluun `silver.silver_games_enriched`, jossa prior-kauden metriikat liitetään avaimella `(season, lineage_id)`.
6. Luo valmiin analyysi- ja koneoppimistaulun `gold.gold_games`.

---

## 4. Datan laadun ja eheyden testaus
dbt ajaa automaattiset skeema- ja eheystarkistukset (Primary Key uniikkius, `not_null`-tarkistukset):
```bash
uv run dbt test
```

Testejä on 47, ja ne kattavat myös joukkuejatkuvuuden: `assert_unique_season_lineage_team`, `assert_prior_metrics_resolved` ja `assert_ari_uta_prior_matches`.

---

## 5. Interaktiivinen dokumentaatio
dbt generoi automaattisen verkkodokumentaation ja lineage-kaavion:
```bash
uv run dbt docs generate
uv run dbt docs serve
```
*Tämä käynnistää paikallisen verkkoselaimen, josta voit tutkia taulujen riippuvuuksia ja sarakkeiden selitteitä.*

---

## 6. Missä data sijaitsee ja miten sitä käytetään ML-mallissa?
Kaikki data tallentuu yhteen jaettavaan DuckDB-tiedostoon:
- **Tiedostopolku:** `data/warehouse/dev.duckdb`
- **Päätuotantotaulu koneoppimiselle:** `gold.gold_games`

### Datan lukeminen Pythonissa (Marimo-notebook tai ML-skripti):
```python
import duckdb

# Yhdistetään tietokantaan
con = duckdb.connect("data/warehouse/dev.duckdb")

# Haetaan valmis aineisto Polars DataFrameksi
df = con.execute("SELECT * FROM gold.gold_games").pl()

print(df.head())
```

---

## 7. Huomioita ja parhaita käytäntöjä kehitykseen
- **Tietokannan lukitukset:** DuckDB sallii useita samanaikaisia lukijoita, mutta vain yhden kirjoittajan kerrallaan. Sulje mahdolliset kirjoitusyhteydet muista skripteistä ennen `dbt run` -ajon suorittamista.
- **Skeemojen nimeäminen:** Projektissa on käytössä makro [`macros/generate_schema_name.sql`](file:///c:/Users/kompl/code/2026/Projektir2/dbt_project/macros/generate_schema_name.sql), joka ohjaa mallit suoraan `silver`- ja `gold`-skeemoihin ilman dbt:n oletusarvoista etuliitettä.
- **Uusien sarakkeiden lisäys:** Jos teet uusia muunnoksia, tee ne Silver-tasolla ja nosta valmis lopputulos Gold-tason `gold_games`-tauluun.
- **Joukkuejatkuvuus (ARI → UTA):** Prior-kauden liitoksissa käytetään `lineage_id`-avainta, ei virallista `team_id`/`team_abbrev`-paria. `silver_team_lineage` mappaa ARI 53 ja UTA 59 → 53. Viralliset tunnisteet säilyvät Gold-taulussa auditointia varten; `home_lineage_id` ja `away_lineage_id` eivät ole ML-piirteitä.
