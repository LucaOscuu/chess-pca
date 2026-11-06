# Pre-registrazione — sasso-carta-forbice su dati umani (11/10/2026, ore 16:45)

Scritta **prima** di stimare la mappa e di calcolare qualunque misura di rotazione sui dati umani. Prima di scriverla ho guardato soltanto: la struttura del file, la coerenza fra scelte e risultati, la ricostruzione dei gruppi e le quote medie di sasso/carta/forbice per gruppo (tutte vicine a un terzo).

## Dati
- Wang, Xu & Zhou (2014), *Scientific Reports* 4: 5830, sottoinsieme distribuito nel pacchetto R `stratEst` 1.1.8 (dataset `WXZ2014`, licenza CC BY 4.0), copia dal mirror GitHub `cran/stratEst`, commit a6373b2.
- 72 studenti, 300 round, gruppi da 6, accoppiamento casuale nel gruppo a ogni round.
- Il file non contiene il gruppo: lo ricostruisco da chi gioca contro chi (per ogni giocatore, i 5 partner con cui le scelte risultano reciprocamente compatibili più spesso). Controllo: in ogni gruppo e in ogni round il multinsieme delle scelte deve coincidere con quello delle scelte "dell'altro" (risultato già verificato: 3.600 su 3.600).
- Il pacchetto non dice quale valore del parametro di vittoria *a* usa questo sottoinsieme (l'esperimento originale usava a = 1,1; 2; 4; 9; 100). Nessuna misura qui sotto usa i payoff.

## Stato e mappa
- Unità: il gruppo (12 traiettorie da 300 round).
- Quote del gruppo per round (r, p, s), lisciate con media mobile **causale** di 10 round (i primi 9 round scartati).
- Vantaggio netto di ogni mossa contro la popolazione: quota di ciò che batte meno quota di ciò che la batte (non dipende da *a*).
- Velocità: differenza delle quote lisciate fra round consecutivi.
- Stato = 9 coordinate (3 quote, 3 vantaggi, 3 velocità), standardizzate; mappa = prime due componenti principali sugli stati di tutti i gruppi insieme (come `rps_sim.py`, senza payoff).
- Nash = (1/3, 1/3, 1/3), velocità zero, proiettato sulla mappa.

## Misure
- **Rotazione sulla mappa:** momento angolare medio attorno alla proiezione di Nash, L = media di (y_t − c) × (y_{t+1} − y_t) sul piano.
- **Rotazione nel simplesso** (controllo della ricostruzione, come WXZ): stessa misura sulle coordinate baricentriche 2D delle quote lisciate, attorno a (1/3, 1/3, 1/3); segno positivo = antiorario R → P → S.
- **Null:** permutazione dell'ordine dei round dentro ogni gruppo (distrugge la dinamica, conserva le quote), applicata alle quote grezze e poi lisciata e proiettata con la stessa mappa; 2.000 permutazioni; p a una coda sul valore assoluto pooled.

## Predizioni
- **H1 (replica di WXZ nel simplesso).** La rotazione è antioraria (R → P → S) in almeno 9 gruppi su 12, e la media pooled supera il 95° percentile del null.
- **H2 (la mappa vede il ciclo).** Sulla mappa la rotazione attorno a Nash ha lo stesso segno in almeno 9 gruppi su 12 e la media pooled supera il 95° percentile del null.
- **H3 (il ciclo non si spegne).** H2 vale anche nella seconda metà (round 151–300) da sola.
- **H4 (dove passa il tempo).** La proiezione di Nash cade in una zona densa della mappa: più del 50% del tempo osservato sta in punti meno densi di Nash. Cioè il gioco umano occupa la zona attorno a Nash (come il gioco stabile simulato) ma vi ruota attorno, a differenza del gioco instabile simulato, dove Nash cade nel centro vuoto.

## Controlli descrittivi (non predizioni)
- A3: autovalori contro un null a spostamento circolare delle quote grezze dentro ogni gruppo (stessa costruzione della mappa).
- A5: angolo fra i piani stimati su due metà dei gruppi (6 contro 6), 200 divisioni casuali.
- Robustezza: lisciatura di 5 e 20 round.

## Cosa si scrive in ogni caso
Le predizioni fallite si dichiarano come tali. Se H1 fallisce, la ricostruzione dei gruppi o il sottoinsieme non riproducono WXZ e le altre ipotesi non si interpretano.
