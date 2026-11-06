# Pre-registrazione — test di trend sulla capacità di calcolo (27/09/2026, 12:40, scritta prima di lanciare)

**Domanda.** Geometria e occupazione cambiano in modo sistematico al crescere della capacità di calcolo dell'agente?
**Capacità (ordinale):** mosse casuali = 0, Stockfish profondità 1 = 1, profondità 3 = 2, profondità 6 = 3. Il corpus Fishtest è escluso: non è "più calcolo", è un processo generativo diverso (controllo di tempo, partite reali).
**Dati:** parquet di `data/cache*` del repo chess-pca. Seed 20260927.

## G — Trend della geometria (finestre di 20 ply)
Per evitare che depth6 sia riferimento e punto della serie insieme, in ogni ripetizione le partite di depth6 si dividono a caso in due metà: **riferimento** e **depth6 di confronto**. Da random, depth1 e depth3 si estrae a caso lo stesso numero di partite della metà di confronto. Per ciascuno dei quattro gruppi si calcola l'angolo fra il suo piano e quello del riferimento. 200 ripetizioni.
Statistica: pendenza β della retta angolo ~ capacità (0, 1, 2, 3).
**Criterio. Il trend regge se:** (i) l'intervallo 2,5–97,5% di β sulle 200 ripetizioni è interamente sotto zero; **e** (ii) in almeno il 95% delle ripetizioni vale angolo(depth1) > angolo(depth3) > angolo(depth6 di confronto). Il confronto con random si riporta ma non è nel criterio: è ovvio che il gioco casuale sia il più lontano.

## O — Trend dell'occupazione (traiettorie intere, materiale appaiato)
Per depth1, depth3 e depth6: gap di occupazione come in `occupancy_grouped.py` (regioni dai quartili del training, scelta sul training, misura sul test), media di Bianco e Nero, media su 5 divisioni. Statistica: pendenza della retta gap ~ capacità (1, 2, 3).
**Null:** le partite dei tre corpora vengono unite e riassegnate a caso a tre gruppi con le numerosità originali, e si ricalcola tutto. 200 permutazioni. p = (b+1)/(m+1).
**Criterio. Il trend regge se** la pendenza osservata è positiva e p < 0,05.

**Se G o O non reggono:** la razionalità limitata resta come motivazione e chiave di lettura; il paper dice che agenti con capacità diverse producono geometrie diverse (già dimostrato), non che la capacità le modella in modo sistematico.
