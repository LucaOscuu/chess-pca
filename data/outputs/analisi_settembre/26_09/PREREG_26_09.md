# Pre-registrazione — 26/09/2026, 14:20 (scritta prima di lanciare)

Dati: parquet di `data/cache*` del repo chess-pca (full = 9.999 partite Fishtest; depth1/3/6, random, 3check = corpora generati). Seed 20260926.

## T1 — Persistenza (A1), definita formalmente

**Definizione.** PCA (2 componenti) stimata sul corpus, senza esiti. Quattro regioni: coda bassa (< q25) e coda alta (> q75) di PC1 e PC2. Una *visita* è una sequenza massimale di ply consecutivi della stessa traiettoria dentro la regione. La *durata* è il numero di ply della visita. Le visite troncate dalla fine della traiettoria sono incluse (analisi principale) ed escluse (sensibilità). La statistica è la durata media delle visite, mediata sulle quattro regioni, calcolata separatamente per Bianco e Nero.

**Null.** Per ogni traiettoria l'ordine dei ply viene rimescolato a blocchi di 10 (block bootstrap), che conserva la dipendenza fra ply vicini. 200 surrogati. Come riferimento permissivo si riporta anche il rimescolamento completo.

**Incertezza.** Intervallo al 95% dell'eccesso relativo (osservato / media dei surrogati − 1) con bootstrap sulle partite (200 ricampionamenti, Bianco e Nero della stessa partita insieme).

**Criterio (corpus full).** La persistenza REGGE se, per entrambe le prospettive, la statistica osservata supera il p95 dei surrogati a blocchi e l'intervallo bootstrap dell'eccesso esclude lo zero. Secondario: stesso test su depth6.

## T2 — A3 con null che conserva la dipendenza

**Null.** Per ogni traiettoria e per ogni coordinata, spostamento circolare di un intervallo casuale, indipendente per coordinata. Ogni coordinata conserva la propria autocorrelazione, mentre i legami fra coordinate si rompono. 200 surrogati.

**Statistica.** Autovalori ordinati della matrice di correlazione, confrontati con il p95 dei surrogati per lo stesso rango (parallel analysis). k* = numero di autovalori consecutivi, a partire dal primo, sopra il p95.

**Criterio.** A3 è SODDISFATTA per un corpus se k* ≥ 2, cioè se il piano intero supera il null. Corpora: full, depth6, random, 3check, su traiettorie intere. Sensibilità: senza la coordinata `material`, che è una combinazione algebrica dei conteggi dei pezzi.

**Limite noto prima di lanciare.** Il null rompe anche i legami algebrici fra coordinate (materiale e conteggi), quindi superarlo misura struttura fra coordinate che include una parte dovuta allo strumento. Da qui la sensibilità senza `material`.

## T3 — Robustezza dell'occupazione

**T3a — molte divisioni (corpus full, materiale appaiato).** Stessa procedura di `occupancy_grouped.py`, ripetuta su 100 divisioni casuali training/test, senza placebo per divisione. REGGE se in almeno il 90% delle divisioni il gap firmato sul test è positivo e supera 1,6 pp (il p95 del placebo del 25/09), per entrambe le prospettive.

**T3b — corpora generati combinati (depth1, depth3, depth6, traiettorie intere, materiale appaiato).** Per ciascun corpus la statistica è la media dei gap firmati di Bianco e Nero sul test, con placebo per partita (1.000 permutazioni, ripetendo la scelta della regione). I p-value dei tre corpora, che hanno partite indipendenti, si combinano con il metodo di Stouffer. REGGE se il p combinato è < 0,05.
