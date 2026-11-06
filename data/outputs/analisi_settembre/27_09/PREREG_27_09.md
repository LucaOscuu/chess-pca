# Pre-registrazione — 27/09/2026, 11:40 (scritta prima di lanciare)

Dati: parquet di `data/cache*` del repo chess-pca. Seed 20260927.

## T3 — A2 con incertezza raggruppata per partita
Stesso calcolo di `a2_partial.py` (tutte le righe, correlazioni grezze e parziali controllando per `material`, esito W = 1, D = 0, L = −1), con intervalli al 95% da bootstrap sulle partite (le due prospettive di una partita restano insieme), 1.000 ricampionamenti, per depth6 (riferimento), antichess e 3check separatamente.
Un'inversione è **robusta** se gli intervalli nel riferimento e nel target escludono entrambi lo zero con segni opposti.
**Criterio.** A2 REGGE se: in antichess almeno 10 inversioni grezze sono robuste; in 3check nessuna è robusta; l'inversione del segno di corr(material, esito) in antichess è robusta. Si riporta anche il conteggio delle inversioni parziali robuste.

## T4 — Contrasti con test appaiato per apertura
Contrasti random, depth1, depth3 e 3check contro depth6 (full è escluso perché non ha aperture comuni). Finestre di 20 ply. Coppie = partite dei due corpora con la stessa posizione di apertura (`opening_fen`); se un'apertura compare più volte, si appaiano in ordine. Permutazione appaiata: per ogni coppia si scambiano le etichette di corpus con probabilità 1/2. 1.000 permutazioni. Statistica = angolo massimo fra i piani. Si ricalcola anche il test non appaiato sulle stesse coppie, per confronto.
**Criterio.** Il test appaiato dà p non maggiori del non appaiato (atteso), e tutti i contrasti restano significativi dopo Holm.

## T5 — Angolo fra i piani di antichess e depth6
Finestre di 12 ply (a 20 restano troppe poche partite), `checks_avail` esclusa da tutti i corpora (varianza nulla in antichess). Contrasti antichess contro depth6 e 3check contro depth6, test di permutazione appaiato per apertura (1.000).
**Ipotesi registrata:** angolo(antichess) > angolo(3check), e antichess significativo. Il confondente di fase (12 ply = apertura) va dichiarato.

## T6 — Leva sul corpus full
Stesso metodo di `feature_leverage.py`: finestre di 20 ply, null split-half raggruppato per partita (50 ripetizioni), 5 randomizzazioni per coordinata.
**Criterio.** Le stesse quattro coordinate (pawns_passed_diff, cnt_pawn, mobility, captures_avail) superano il contrasto massimo fra sistemi (33,4°), oppure restano le prime quattro per leva.

## T2 — Robustezza al metodo di compressione (analisi fattoriale al posto della PCA)
Analisi fattoriale a 2 fattori (`sklearn.decomposition.FactorAnalysis`, senza rotazione; conta il sottospazio, non gli assi) su dati standardizzati.
- (a) Contrasto random contro depth6, 20 ply, permutazione appaiata 500 volte: **regge** se p < 0,05.
- (b) Leva su depth6, 20 ply: **regge** se le 4 coordinate a leva più alta con la PCA sono fra le prime 6 con l'analisi fattoriale.
- (c) Occupazione sul corpus full a materiale appaiato, 20 divisioni, regioni dai quartili dei punteggi fattoriali del training: **regge** se ≥ 90% delle divisioni supera 1,6 pp per W e per B.
