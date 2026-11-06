# Pre-registrazione — differenza fra AUC (imprese), 27/09/2026 12:20, scritta prima di lanciare

**Contesto.** Il test T3 del 24/09 ha dato AUC di fallimento: direzione condivisa fra strumenti 0,610; seconda direzione, non condivisa, 0,693; PC1 della mappa 1 0,696. Gli intervalli erano calcolati separatamente, quindi non dicono se le differenze sono nette.

**Test.** Stessa procedura e stessi dati di `run_consolidation.py` (split A/B con seed 20260925; punteggi medi per impresa sul lato B). Bootstrap appaiato per impresa sul lato B (2.000 ricampionamenti): in ogni ricampionamento si calcolano entrambe le AUC sulle stesse imprese e se ne fa la differenza. Orientamento di ogni punteggio fissato sul campione completo, come nell'originale.
- Δ1 = AUC(seconda direzione, non condivisa) − AUC(direzione condivisa)
- Δ2 = AUC(PC1 mappa 1) − AUC(direzione condivisa)

**Criterio.** "Prevedere ≠ attribuire" è sostenuto se l'intervallo al 95% esclude lo zero sia per Δ1 sia per Δ2.

**Secondario (stabilità).** Ripetere il calcolo puntuale di Δ1 e Δ2 su 20 split A/B diversi. Si riportano mediana e quota di split con Δ > 0.
