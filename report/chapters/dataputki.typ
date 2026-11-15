= Dataputki ja tiedon esikäsittely

== Medallion-arkkitehtuurin toteutus dbt:llä ja DuckDB:llä

Projektin tiedonhallinnan ja esikäsittelyn perustaksi valittiin medallion-arkkitehtuuri, joka jakaa datan kolmeen laadulliseen kerrokseen: pronssi (Bronze), hopea (Silver) ja kulta (Gold). Arkkitehtuurin toteutuksessa käytetään dbt-työkalua (`dbt-duckdb`) yhdistettynä sulautettuun ja suorituskykyiseen DuckDB-saraketietokantaan.

Ratkaisun keskeiset hyödyt projektin kannalta ovat:
- *Laadunhallinnan eriyttäminen:* Raakadata säilyy muuttumattomana pronssitasolla, jolloin prosessointilogiikan muuttuessa dataa ei tarvitse noutaa uudelleen ulkoisesta rajapinnasta.
- *Nopeus ja keveys:* DuckDB on sarakeorientoitunut OLAP-tietokanta, joka mahdollistaa aggregoinnit ja raskaat ikkunafunktiot ilman erillistä tietokantapalvelinta tai pilvi-infrastruktuuria. Koko kauden käsittely dbt:llä vie vain noin sekunnin.
- *Yhtenäinen lähde totuudelle (Single Source of Truth):* Kaikki laskennat (kuten lepopäivät ja kuntopuntarit) on keskitetty SQL-malleihin sen sijaan, että kukin koneoppimismallia tutkiva tiimiläinen tekisi erillisiä ja toisistaan poikkeavia ad hoc -skriptejä.

== Kerrosten roolit ja tietovirta

=== Pronssitaso (Bronze - Raakadata)
Pronssitaso vastaa alkuperäisen datan tuonnista ja tallentamisesta sellaisenaan. Tiedonhaku NHL:n virallisista REST-rajapinnoista (`api-web.nhle.com` ja `api.nhle.com/stats/rest`) suoritetaan Pythonilla (`src/bronze_ingestion.py`), joka tallentaa raakadata-taulut DuckDB:hen. Tämän jälkeen dbt:n pronssikerroksen mallit ottavat aineiston hallintaansa:
1. `bronze_games`: Alkuperäinen otteludata sisältäen kaikki ottelutapahtumat, aikaleimat ja raa'at JSON-vastaukset.
2. `bronze_team_stats`: Alkuperäiset joukkuetilastot per kausi.
3. `bronze_team_percentages`: Alkuperäiset kehittyneet hallintatilastot (Corsi, Fenwick ja PDO).

=== Hopeataso (Silver - Puhdistus, tulosliput ja rikastus)
Hopeatasolla raakadata puhdistetaan, standardoidaan ja rikastetaan koneoppimista varten:
1. `stg_games`: Muuntaa tekstimuotoiset päivämäärät `DATE`-tyypiksi, poistaa mahdolliset duplikaatit ja muodostaa selkeät *tulosliput*.
2. `stg_team_stats` ja `stg_team_percentages`: Puhdistavat ja pyöristävät kausitilastot.
3. `silver_team_metrics`: Yhdistää perinteiset tilastot ja edistyneet laukaisutilastot yhdeksi joukkueprofiiliksi.
4. `silver_games_enriched`: Laskee kullekin ottelulle joukkueiden lepopäivät, back-to-back -ottelut sekä kuntopuntarin.

==== Tulosliput (Result Flags) ja niiden merkitys
NHL:n API-rajapinnasta saapuvassa datassa otteluiden tilat ja lopputulokset ovat monimutkaisessa ja osin epäyhtenäisessä muodossa (esim. `gameState: "FINAL"`, `"OFF"` tai `"FUT"`). Hopeatasolla dataan luodaan selkeät ja standardoidut tulosliput (binääriset tai loogiset indikaattorimuuttujat):
- *`is_played` (TRUE / FALSE):* Kertoo, onko ottelu jo pelattu loppuun vai onko kyseessä vasta otteluohjelmassa oleva tuleva peli (`FUT`). Tämä on kriittinen suodatin, jottei tulevien otteluiden tyhjiä arvoja (`NULL`) syötetä mallin opetusdataan.
- *`home_win` ja `away_win` (1 tai 0):* Yksiselitteinen numeerinen lopputulos (1 = voitto, 0 = tappio). `home_win` toimii koneoppimisessa luokittelumallin keskeisenä *ennustekohteena* (Target variable).
- *`is_overtime` ja `is_regulation` (TRUE / FALSE):* Erottelee, päättyikö ottelu varsinaisella peliajalla vai jatkoajalla/voittomaalikilpailussa. Jääkiekossa tasapeli varsinaisen peliajan jälkeen jakaa sarjapisteitä eri tavalla, joten tieto on olennainen lajianalyysissä.
- *`total_goals` ja `goal_diff`:* Ottelun yhteenlaskettu maalimäärä ja maaliero.

==== Datavuodon (Data Leakage) matemaattinen esto ikkunafunktioilla
Koneoppimisessa aikasarja- ja urheiludatan yleisimpiä virheitä on *datavuoto* (data leakage), jossa mallin opetusvaiheessa piirteisiin päätyy tietoa tulevaisuudesta tai kyseisen ottelun lopputuloksesta. Jos esimerkiksi joukkueen viimeaikaista maalitahtia laskettaisiin tavallisella liukuvalla keskiarvolla ottaen huomioon myös kyseisen ottelun tehdyt maalit, malli saisi tietoonsa ottelun lopputuloksen jo ennen ennustamista. Tällöin mallin tarkkuus näyttäisi opetusvaiheessa valheellisen hyvältä, mutta romahtaisi oikeassa käytössä tulevia otteluita ennustettaessa.

Datavuoto estetään hopeatasolla (`silver_games_enriched`) käyttämällä SQL-ikkunointia, jossa laskentaikkuna päätetään tiukasti yhtä peliä ennen kyseistä ottelua:
```sql
AVG(goals_for) OVER (
    PARTITION BY season, team_abbrev
    ORDER BY game_date, game_id
    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
) AS rolling_goals_for_5
```
Määritys `ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING` tarkoittaa:
1. Laskenta alkaa 5 ottelua taaksepäin (`5 PRECEDING`).
2. Laskenta päättyy tarkalleen yhtä ottelua ennen (`1 PRECEDING`).
3. Nykyinen rivi (`CURRENT ROW`) eli ennustettava ottelu *ei ole lainkaan mukana laskennassa*.

Näin varmistetaan matemaattinen ja kronologinen eheys: otteluhetkellä mallille tarjottavat kuntopuntarin piirteet (voittoprosentti, tehdyt maalit ja päästetyt maalit viimeisimmästä 5 ja 10 ottelusta) heijastavat tarkasti vain sitä tietoa, joka oli saatavilla ennen ottelun alkamista.

==== Lepopäivät ja rasitustekijät
Ikkunafunktioilla lasketaan myös kummankin joukkueen lepopäivät edellisestä ottelusta:
```sql
date_diff('day', LAG(game_date) OVER (...), game_date) AS days_rest
```
Jos lepopäivien määrä on tasan 1, kyseessä on peräkkäisinä päivinä pelaaminen eli ns. *back-to-back* -ottelu (`is_b2b = 1`). Jääkiekkoanalyysissä lepoaika ja matkustusrasitus ovat todistetusti merkittäviä ottelutulokseen vaikuttavia tekijöitä.

=== Kultataso (Gold - Koneoppimisvalmis data)
Kultatason taulu `gold.gold_games` muodostaa selkeän ja riippumattoman rajapinnan koneoppimisen baseline-mallille. Se yhdistää ottelutiedot, standardoidut tulosliput (Targets), lepoajat, kuntopuntarit sekä joukkueiden väliset differentiaalit (koti vs. vieras erotukset) yhteen ainoaan tauluun.

== Toteutuksen kriittinen arviointi

=== Onnistumiset ja vahvuudet
- *Automaattinen testaus:* dbt:n kautta määritellyt 30 skeema- ja datatestiä (`unique`, `not_null`) varmistavat aineiston eheyden automaattisesti jokaisella ajolla kaikissa kerroksissa.
- *Suorituskyky:* Dataputki kykenee prosessoimaan 1 500 ottelua ja useita kausia sekunneissa, mikä nopeuttaa kokeilujen tekemistä.
- *Koneoppimisen työnjaon selkeys:* Ennustemalleja rakentavan tiimiläisen ei tarvitse huolehtia SQL-liitoksista tai datan puhdistamisesta, vaan hän voi ladata suoraan analyysivalmiin taulun.

=== Havaitut haasteet ja rajoitteet
- *API:n dokumentoimattomuus:* NHL:n rajapinnat ovat epävirallisia ja voivat muuttua ilman ennakkovaroitusta. Lisäksi rajapinnasta puuttuvat loukkaantumis- ja kokoonpanotiedot.
- *DuckDB-tiedostolukitus:* DuckDB sallii kerrallaan vain yhden kirjoittajan, mikä vaatii huolellisuutta yhteyksien sulkemisessa interaktiivisissa notebookeissa.
- *Kausitilastojen staattisuus:* `bronze.raw_team_stats` sisältää koko kauden aggregoidut tilastot, jolloin alkukauden otteluissa on mukana kauden kokonaistilastoihin perustuvaa tietoa.

=== Jatkokehitysehdotukset
1. *Kumulatiivinen sarjataulukkolaskenta:* Sarjataulukon pisteet ja maalisuhde voidaan jatkossa laskea dbt:ssä kumulatiivisesti vain ennen kyseistä päivää pelatuista otteluista.
2. *Pelaaja- ja maalivahtitilastojen liittäminen:* Aloittavan maalivahdin torjuntaprosentti ja kärkipelaajien pistetahti rikastaisivat ennustepotentiaalia merkittävästi.
3. *Automatisoitu ajastus:* Dataputken ajo voitaisiin automatisoida esimerkiksi GitHub Actions -työnkululla kerran vuorokaudessa.
