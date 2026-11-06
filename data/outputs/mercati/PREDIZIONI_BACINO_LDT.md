# Predizioni registrate prima del test — bacino e abilità, learning direction (04/10/2026, 16:15)

**Dati e feature.** Gli stessi di `PREDIZIONI_MERCATI.md`: 166 mercati, 3.088 traiettorie, set "completo" e set "solo comportamento".

## Test B — Bacino dei vincitori e abilità cognitiva (Corgnet, DeSantis & Porter 2018)

**Costruzione del bacino (Empirical Attractor Basin)**, con divisione casuale dei mercati in due metà:
- **Training.** Su metà dei mercati si stimano PCA e quartili. Fra le 4 regioni candidate (coda bassa o alta di PC1 o PC2), il bacino è quella in cui chi chiude sopra la mediana passa più tempo, rispetto a chi chiude sotto.
- **Test.** Sull'altra metà si misura, per ogni traiettoria, la quota di periodi passati nel bacino.

**Misure.** Correlazione di Spearman fra abilità cognitiva (`skills`) e quota nel bacino.
- Il null permuta le abilità fra i trader dello stesso mercato (1.000 permutazioni).
- Si ripete su 20 divisioni casuali dei mercati e si riporta la mediana.

**Predizioni.**
- **B1.** ρ > 0 con p ≤ 0,05: i più abili occupano di più il bacino.
- **B2 (esplorativa, nessuna soglia).** In una regressione del guadagno relativo su abilità e quota nel bacino, il coefficiente dell'abilità si riduce quando si aggiunge il bacino. Il bacino spiegherebbe parte del legame abilità → guadagno.

## Test C — Learning direction (Selten & Neugebauer 2019)

- **Errore ex post al periodo t.** Il trader ha comprato a un prezzo medio sopra il valore fondamentale (buy_q > 0 e buy_dev > 0,1), oppure ha venduto sotto (sell_q > 0 e sell_dev < −0,1).
- **Direzione vincente u.** Sui mercati di training: posizione media di chi chiude sopra la mediana meno posizione media di chi chiude sotto, nel piano PC1–PC2, normalizzata.
- **Risposta.** Il passo t → t+1 proiettato su u, confrontato con i passi dopo periodi senza errore (media per trader, poi fra trader).

**Predizioni.**
- **C1 (controllo diretto, senza mappa).** Dopo aver pagato troppo, il prezzo dei bid al periodo t+1 scende rispetto a t più che senza errore.
- **C2.** Dopo un errore, la proiezione del passo su u è maggiore che senza errore (p ≤ 0,05, permutazione per trader).
- **C3.** La risposta C2 è più forte nei trader ad alta abilità.

## Smentite
- **B1 falsa:** il bacino non è legato all'abilità cognitiva.
- **C2 falsa:** sulla mappa non si vede la learning direction.
