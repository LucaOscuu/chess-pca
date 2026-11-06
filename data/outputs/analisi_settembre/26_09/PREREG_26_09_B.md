# Pre-registrazione B — 26/09/2026, 15:45 (scritta prima di lanciare)

## B1 — Occupazione con strumenti diversi (scacchi, corpus full, materiale appaiato)

Strumenti, cioè varianti del feature map (d varia):
- I0: tutte le 18 coordinate (riferimento)
- I1: senza le 4 coordinate a leva alta (pawns_passed_diff, cnt_pawn, mobility, captures_avail), 14 coordinate
- I2: solo la famiglia del materiale (material, cnt_pawn, cnt_knight, cnt_bishop, cnt_rook, cnt_queen, pawns_passed_diff, hanging_diff), 8 coordinate
- I3: solo coordinate posizionali (le altre 10: mobility, captures_avail, checks_avail, attack_overlap, center_core, center_ext, king_safety_diff, pawns_doubled_diff, pawns_isolated_diff, delta_mobility)

Procedura identica a `occupancy_grouped.py` (regioni dai quartili del training, scelta della regione sul training, misura sul test, materiale appaiato |material| ≤ 50). 20 divisioni per strumento, più una corsa con placebo per partita (1.000 permutazioni) sulla prima divisione.

**Criterio per strumento:** regge se in ≥ 90% delle 20 divisioni il gap firmato supera 1,6 pp per W e per B, e il placebo dà p < 0,05 per entrambi.

**Accordo fra strumenti:** sul test della prima divisione, correlazione di Spearman fra le residence share per traiettoria di I0 e di ciascun Ik (regione scelta da ciascuno strumento), con il segno orientato in modo che valori alti indichino vittoria.

**Affermazione 3 negli scacchi:** SOSTENUTA se l'occupazione regge con I1, I2 e I3; PARZIALE se regge con due su tre; NON SOSTENUTA altrimenti.

## B2 — Scomposizione della variabilità della geometria (PERMANOVA)

Finestre di 20 ply (come i contrasti del report). Distanza fra geometrie = distanza di proiezione ‖P_A − P_B‖_F², con P = V Vᵀ (piano delle prime due componenti, d = 18). Le proiezioni vettorizzate stanno in uno spazio euclideo, quindi la scomposizione è quella di una MANOVA sulle proiezioni: SS totale = SS sistema + SS strumento + SS interazione + SS residuo (campionamento). Si riportano le quote.

- **Disegno A (distruttivo):** 6 sistemi (full, depth1, depth3, depth6, random, 3check) × 19 strumenti (originale + 18 con una coordinata randomizzata marginalmente) × 5 metà casuali per partita.
- **Disegno B (trasformazioni monotone, "cambio di unità"):** 6 sistemi × 3 strumenti (originale, ranghi per coordinata, log con segno) × 5 metà casuali.

**Ipotesi registrate:**
- A: quota strumento > quota sistema.
- B: quota strumento < quota sistema. Le trasformazioni monotone assomigliano a cambi di unità e dovrebbero pesare poco. Questo è il test negli scacchi dell'affermazione 1 ("unità no").

**Cautela da riportare:** le quote dipendono dal disegno, cioè da quali sistemi e quali varianti dello strumento si includono.
