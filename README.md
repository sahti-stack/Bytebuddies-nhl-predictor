# Bytebuddies NHL Predictor

**KAMK Student Project 2 – Machine Learning**

Tämä repositorio sisältää koneoppimisprojektin dataputken, tutkimusmuistiot (Marimo), koneoppimismallit sekä opintojakson dokumentaation ja raportin. 

Projekti käyttää monoreporakennetta, jota hallitaan **uv**-työkalulla Python-riippuvuuksille, **dbt:llä** ja **DuckDB:llä** Medallion-dataputkelle (Bronze $\rightarrow$ Silver $\rightarrow$ Gold), **Marimolla** reaktiivisille muistioille sekä **Typstillä** (`kamk-thesis`) raportoinnille.

---

## Projektirakenne

```text
Bytebuddies-nhl-predictor/
├── .venv/               # Virtual environment managed by uv (ignored by Git)
├── data/                # Paikallinen DuckDB-tietokanta ja raakadata (ignored by Git)
├── dbt_project/         # dbt Medallion -dataputki (DuckDB)
│   ├── macros/          # Mukautetut Jinja-makrot (mm. skeeman nimeäminen)
│   └── models/          # Medallion-kerrokset:
│       ├── bronze/      # 🥉 Raakadatataulut suoraan API-lähteistä
│       ├── silver/      # 🥈 Puhdistus, tulosliput, lepopäivät ja kuntopuntarit
│       └── gold/        # 🥇 Koneoppimisvalmis data (gold_games)
├── docs/                # Tiimin arkkitehtuuri- ja käyttöohjeet
│   ├── dbt-ohje.md      # Yksityiskohtainen dbt-käyttöohje
│   └── dbt_lineage_kaavio.md # Datan sukupuu, arkkitehtuurianalyysi ja perustelut
├── notebooks/           # Marimo reaktiiviset tutkimusmuistiot (.py)
├── report/              # Typst-raporttipohja (KAMK thesis)
│   ├── chapters/        # Raportin luvut (mm. projektisuunnitelma ja dataputki)
│   └── thesis.typ       # Raportin päädokumentti
├── src/                 # Dataputken Python-lähdekoodit
│   └── bronze_ingestion.py # NHL API -tiedonkeruu DuckDB:n Bronze-kerrokseen
├── pyproject.toml       # Projektin riippuvuudet ja työkalut
├── README.md            # Tämä pääohjeistus
└── uv.lock              # Lukitut pakettiversiot
```

---

## Esivaatimukset

Varmista, että koneellesi on asennettu:

### 1. `uv` (Python-pakettienhallinta)
* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **Linux / macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Carlito-fontti (raportin kääntämistä varten)
KAMK Typst -pohja vaatii **Carlito**-fonttiperheen.
* **Lataa:** [Google Fonts: Carlito](https://fonts.google.com/specimen/Carlito) ja asenna `.ttf`-tiedostot järjestelmään.

### 3. Typst (raportointia varten)
* **VS Code:** Asenna **Tinymist Typst** -laajennus.
* **CLI:** `winget install --id Typst.Typst` (Windows) tai `brew install typst` (macOS).

---

## Alkuasetukset (Initial Setup)

1. **Kloonaa repositorio ja siirry kansioon:**
   ```bash
   git clone https://github.com/sahti-stack/Bytebuddies-nhl-predictor.git
   cd Bytebuddies-nhl-predictor
   ```

2. **Synkronoi Python-ympäristö:**
   Komento asentaa automaattisesti oikean Python-version ja kaikki lukitut paketit:
   ```bash
   uv sync
   ```

---

## 🏒 Dataputken ajaminen (dbt Medallion Pipeline)

Dataputki noutaa tuoreen NHL-datan, puhdistaa sen ja tuottaa valmiin koneoppimisen syöteaineiston (`gold.gold_games`) paikalliseen DuckDB-tietokantaan.

### Vaihe 1: Datan haku rajapinnasta (Bronze Ingestion)
Suorita tiedonkeruu projektin juuresta. Skripti noutaa ottelut, sarjataulukon ja edistyneet Corsi/Fenwick/PDO-tilastot:
```bash
# Esimerkki: Haetaan kausi 2024–2025
uv run python src/bronze_ingestion.py --seasons 20242025

# Esimerkki: Useampi kausi kerralla
uv run python src/bronze_ingestion.py --seasons 20222023 20232024 20242025
```

### Vaihe 2: dbt-muunnokset (Bronze ➔ Silver ➔ Gold)
Siirry `dbt_project`-kansioon ja aja mallit:
```bash
cd dbt_project

# Tarkista tietokantayhteys
uv run dbt debug

# Aja kaikki Medallion-taulut (9 mallia)
uv run dbt run
```

### Vaihe 3: Datan laadun ja eheyden testaus
Aja dbt:n automaattiset laatutestit (varmistaa pääavainten uniikkiuden ja kenttien eheyden):
```bash
uv run dbt test
```

### Vaihe 4: Interaktiivinen dokumentaatio ja Lineage-kaavio
```bash
uv run dbt docs generate
uv run dbt docs serve
```
*Avaa selaimeen visuaalisen esityksen tietokannan rakenteesta ja riippuvuussuhteista.*

### Vaihe 5: Valmiin datan lukeminen koneoppimisessa
Koneoppimismallia kehittävä voi lukea valmiin aineiston suoraan DuckDB:stä Polars-muodossa:
```python
import duckdb

con = duckdb.connect("data/warehouse/dev.duckdb")
df = con.execute("SELECT * FROM gold.gold_games").pl()
print(df.head())
```

Lisätietoja:
* 📖 [Yksityiskohtainen dbt-ohje (`docs/dbt-ohje.md`)](docs/dbt-ohje.md)
* 📊 [Arkkitehtuuri, riippuvuussukupuu ja analyysi (`docs/dbt_lineage_kaavio.md`)](docs/dbt_lineage_kaavio.md)

---

## Muiden työkalujen ajaminen

### 1. Marimo-tutkimusmuistiot (`notebooks/`)
* **Avaa Marimo-kojelauta:**
  ```bash
  uv run marimo edit
  ```
* **Muokkaa tiettyä muistiota:**
  ```bash
  uv run marimo edit notebooks/panu/nhl_api_exploration.py
  ```

### 2. Raportin kääntäminen (`report/`)
* **Käännä raportti PDF-muotoon:**
  ```bash
  typst compile report/thesis.typ report/thesis.pdf
  ```
* **Katselutila automaattipäivityksellä:**
  ```bash
  typst watch report/thesis.typ --open
  ```

---

## Uusien pakettien lisääminen
Uudet riippuvuudet asennetaan aina `uv`:n kautta projektin juuresta:
```bash
uv add <paketin-nimi>
```
