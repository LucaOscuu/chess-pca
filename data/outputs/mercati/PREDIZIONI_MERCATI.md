# Predizioni registrate prima del test — mercati sperimentali (04/10/2026, pomeriggio)

**Dati.** Replica preregistrata di Huber, Holzmeister, Johannesson, König-Kersting, Dreber, Huber & Kirchler (2024): 166 mercati con asta doppia continua, 1.544 trader, 2 ripetizioni da 10 periodi. Il valore fondamentale scende da 55 a 5. Dal kit OSF.

**Traiettoria.** Un trader in una ripetizione (3.088 traiettorie, 10 periodi ciascuna).

**Feature per trader e periodo** (13; tutte le grandezze di stock sono relative alla media del mercato):
- **posizione:** azioni detenute, cassa, ricchezza al valore fondamentale;
- **ordini:** numero di bid, numero di ask, quota di ordini al mercato, quota di cancellazioni, prezzo medio dei bid e degli ask rispetto al fondamentale (log);
- **scambi:** quantità comprata e venduta, prezzo medio degli acquisti e delle vendite rispetto al fondamentale (log).

**Esito.** Guadagno finale sopra la mediana del proprio mercato (vince) o sotto (perde).

**Analogo del "materiale".** La ricchezza al valore fondamentale. L'appaiamento si fa con wealth_rel in [0,9; 1,1].

**Robustezza.** Il set "solo comportamento" toglie ricchezza, cassa e prezzi realizzati, che sono vicini all'esito per costruzione.

## Test e predizioni
- **T0 (esiste una geometria?).** Lo spettro della PCA deve superare il null ottenuto mescolando ogni colonna (λ1 e λ2 sopra il p95 del null). Il piano PC1–PC2 deve essere stabile su metà dei mercati scelte a caso (angolo mediano split-half < 30°).
- **T1 (la mappa è del mercato).** Due confronti: ripetizione 1 vs 2 (esperienza) e trader ad alta vs bassa abilità cognitiva (mediana di `skills`). Predizione: l'angolo fra i piani **non supera** il null permutazionale (p > 0,05), come per umani e Stockfish negli scacchi.
- **T2 (occupazione).** Le regioni sono i quartili di PC1 e PC2, stimati su metà dei mercati senza guardare gli esiti. Gli esiti si usano solo per scegliere la regione. Il gap si misura sull'altra metà, a ricchezza appaiata. Il placebo permuta gli esiti dentro ogni mercato. Predizione: gap > 0 con p ≤ 0,05.
- **T3 (flusso e razionalità limitata).** Il campo di velocità dei trader ad alta abilità differisce da quello dei trader a bassa abilità (p ≤ 0,05), anche se la mappa è la stessa (T1). Stesso confronto fra ripetizione 1 e 2.

## Smentite
- **T0 fallito:** nei mercati non c'è una geometria stabile, e il framework non si trasferisce.
- **T1 fallito:** la mappa dipende da chi gioca, non dalle regole.
- **T2 o T3 falliti:** la mappa esiste ma non dice nulla di economico.
