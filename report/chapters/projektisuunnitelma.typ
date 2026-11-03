= Projektisuunnitelma

== Projektin määrittely

Projektin tavoitteena on prosessoida ja löytää tuoreesta NHL:n raakadatasta koneoppimisen avulla pelitapahtumista riippuvuuksia, joita on vaikea havaita. Tiedonkeruussa käytämme `nhl-api-py` 3.6.0 -kirjastoa, joka käyttää NHL:n virallisia API-rajapintoja.

Ensimmäisessä vaiheessa rakennamme dataputken ja mallin, joka pyrkii ennustamaan koti- ja vieraspelin tuloksia sekä maalimääriä. Tämä toimii proof-of-concept- ja baseline-mallina, josta voimme todeta, että dataputkemme toimii alusta loppuun saakka. Lopullinen ennustuskohde lyödään lukkoon tämän vaiheen päätyttyä, kun dataputki on todettu toimivaksi ja olemme tutustuneet dataan riittävän syvällisesti.

Datan tutkimiseen käytämme Marimo-notebookeja, joiden avulla etsimme datasta sellaisia piirteitä, jotka auttavat mallin opettamisessa.

Dataputken rakennamme dbt:n päälle, koska kaikilla on kokemusta siitä ja se toimii erinomaisesti tällaisessa koneoppimisprojektissa. Käytämme medallion-arkkitehtuuria:

- *Pronssitasolle* tuomme raakadatan API-rajapinnoista itsellemme.
- *Hopeatasolla* siivoamme ja rikastamme dataa piirremuokkauksen keinoin, jotta se oppii aineistosta huomattavasti raakadataa syvemmin.
- *Kultatasolla* meillä on valmiiksi käsitelty data, joka voidaan syöttää koneoppimismalleille.

== Aikataulu ja toteutus

*Tärkeät päivämäärät ja välitavoitteet:*

- *2.11.2026:* Projektisuunnitelman esitys parillisille tiimeille (parillisuusehto: Kakkosen jakojäännös tiiminumerosta on nolla)
- *16.11.2026:* Sprint-esitys 1
- *30.11.2026:* Sprint-esitys 2
- *14.12.2026:* Loppuesitykset (\~10 tiimiä, 15 min/esitys)
- *31.12.2026:* Deadline kaikille palautuksille viimeistään klo 23:59

*Sprinttiaikataulu:*

- *Sprint 0 (29.10.-4.11.):* Tutkitaan dataa ja loihditaan tämä artefakti kohdilleen. Jiran valmistelu. Kehitysympäristön valmistelu.
- *Sprint 1 (5.11.-11.11.):* Datan EDA ja MVP valmiiksi
- *Sprint 2 (12.11.-18.11.):* Datan valmistelu lopulliseen muotoonsa
- *Sprint 3 (19.11.-25.11.):* Eri mallien lopputulosten vertailu ja mallien hienosäätäminen
- *Sprint 4 (26.11.-2.12.):* Frontti
- *Sprint 5 (3.12.-9.12.):* Loppuviimeistelyt (Sprinttejä saatetaan lisätä tarpeen vaatiessa)

Projekti toteutetaan Scrum-viitekehyksen mukaisesti. Sprinttien pituudet määriteltiin viikon pituisiksi. Sprintit alkavat torstaisin ja loppuvat keskiviikkoisin katselmoinnin ja retrospektiivin jälkeen uuden sprintin suunnitteluun. Dailyja ainakin maanantaisin.

== Resurssit ja viestintä

Resursseiksi on määritelty 5 opintopisteen mukainen 135 tuntia per opiskelija, josta minimissään 70 %, \~95 h.

Viestintään käytetään pääsääntöisesti Discordiin tarkoitukseen perustettua kanavaa.

== Riskienhallinta

Riskienhallinta on olennainen osa jokaista ohjelmistokehitys- ja koneoppimisprojektia. Tässä osiossa käymme läpi keskeisimmät tunnistetut riskit NHL-ennustusprojektissamme sekä suunnitellut toimenpiteet niiden hallitsemiseksi.

=== API-riski - Epävirallisen rajapinnan käyttö

NHL:n julkinen API on epävirallinen ja dokumentoimaton, eli NHL voi sulkea tai muuttaa sen milloin tahansa ilman ennakkovaroitusta. Tämä on projektin merkittävin tekninen riski.

*Hallintakeino:* Kaikki haettava data tallennetaan paikalliseen DuckDB-tietokantaan heti ensimmäisen haun jälkeen. Näin projekti ei ole riippuvainen API:n jatkuvasta saatavuudesta, ja data säilyy vaikka rajapinta muuttuisi.

=== Puuttuva tieto - Loukkaantumiset ja kokoonpanot

NHL:n julkisessa APIssa ei ole loukkaantumistietoja eikä virallisia kokoonpanotietoja ennen ottelua. Tämä on merkittävä puute ennustemallin näkökulmasta, sillä pelaajien kunto vaikuttaa ottelutuloksiin.

*Hallintakeino:* Tunnistamme ja dokumentoimme rajoituksen avoimesti. Loppuraportissa kerromme selkeästi, että malli ei pysty ottamaan loukkaantumisia huomioon. Tämä on realistinen ja rehellinen tapa käsitellä ongelmaa. Tulevaisuudessa voisi harkita kolmannen osapuolen lähteiden käyttöä.

=== Malliriski - Ylisovittaminen (Overfitting)

Koneoppimismalli voi "oppia ulkoa" yhden kauden erityispiirteet sen sijaan, että se oppii yleistettäviä lainalaisuuksia. Tämä johtaa tilanteeseen, jossa malli toimii hyvin koulutusdatajoukolla mutta huonosti oikeilla otteluilla.

*Hallintakeino:* Käytämme ns. cross-season-validointia, jossa koulutamme mallin esim. kausilla 2018-2023 ja testaamme sen kausi 2023-2024 -datalla. Tämä on oikea tapa mitata mallin todellista yleistymiskykyä.

=== Datariski - Luokkien epätasapaino

NHL:ssä kotijoukkue voittaa tilastollisesti useammin kuin vierasjoukkue. Jos malli oppii tämän harhan, se voi ennustaa kotijoukkueen voiton lähes aina, mikä näyttää hyvältä tilastoissa mutta ei ole hyödyllinen ennuste.

*Hallintakeino:* Käytämme mallien arviointiin tarkoituksenmukaisia mittareita kuten F1-scorea ja ROC-AUC:ta pelkän tarkkuuden (accuracy) sijaan. Tarvittaessa tasapainottamme luokkia painotuksilla (class weights).

=== Projektiriski - Laajuuden hallinta

Koneoppimisprojekteissa on tyypillistä, että uusia ideoita ja ominaisuuksia tulee mieleen koko ajan. Ilman selkeää priorisointia projektin laajuus kasvaa hallitsemattomasti.

*Hallintakeino:* Käytämme Jira-projektihallintatyökalua sprinttien hallintaan. Kaikki uudet ideat lisätään backlogiin, mutta toteutettavat tehtävät päätetään sprint-suunnittelussa. Näin pysymme aikataulussa ja fokus pysyy oikeissa asioissa.

=== Eettinen riski - Ennustetyökalun väärinkäyttö

Ottelutuloksia ennustava malli voidaan yhdistää uhkapelaamiseen. Vaikka projekti on opetustarkoituksessa, on tärkeää tiedostaa tämä riski.

*Hallintakeino:* Loppuraportissa ja mahdollisessa sovelluksessa lisätään selkeä disclaimer, jossa todetaan, että kyseessä on oppimisprojekti eikä vedonlyöntisuositus. Emme integroi projektia mihinkään vedonlyöntipalveluun.

=== Yhteenveto

Olemme tunnistaneet kuusi keskeistä riskiä: API-riski, puuttuva tieto, ylisovittaminen, luokkien epätasapaino, laajuuden hallinta sekä eettinen vastuu. Jokaiselle riskille on suunniteltu hallintakeino. Suurin avoin riski on loukkaantumistietojen puuttuminen - sen hyväksymme ja dokumentoimme avoimesti sen sijaan, että yrittäisimme piilottaa sen.
