# Predizioni registrate prima dell'analisi — umani vs motore (03/10/2026)

Scritte prima di aver estratto le feature dalle partite umane e prima di qualsiasi numero.
Non si modificano dopo aver visto i risultati.

## Dati
- **Umani classica**: partite OTB a cadenza lunga trasmesse su Lichess Broadcast, entrambi i giocatori con Elo FIDE >= 2400.
- **Umani veloci**: partite rapid e blitz dalla stessa fonte, stessa soglia Elo. La cadenza si ricava dall'header TimeControl, altrimenti dalla categoria FIDE del torneo.
- **Motore**: i corpora già in cache, cioè depth6 (2.000 partite, aperture fisse) e full (Fishtest).
- **Feature map**: `features.py` invariato (18 coordinate). Pipeline invariata.

## Finestre
- **Primaria**: righe 10–29 per prospettiva, circa ply 21–60, cioè dopo l'apertura. Serve a togliere l'effetto delle aperture fisse dei corpora del motore.
- **Secondaria**: righe 0–19, la finestra usata nel REPORT v4, per confrontabilità.

## Test
1. **Rotazione del piano.** Si misura l'angolo principale massimo fra i piani PC1–PC2. Il null è permutazionale, raggruppato per partita, a numerosità appaiate e con 1.000 permutazioni (stesso codice di `permutation_contrasts.py`).
2. **Occupazione.** Si usa `occupancy_grouped.py` senza modifiche: regioni dai quartili del training, gap fuori campione a materiale appaiato (|material| <= 50 cp), placebo per partita, 1.000 permutazioni.
3. **Obiettivo.** Si calcola la correlazione parziale delle coordinate con l'esito, controllando per il materiale, come nel test A2. Si contano le inversioni di segno rispetto a depth6 con |r| > 0,05 in entrambi.

## Predizioni
- **P1.** Umani classica vs depth6: l'angolo supera il null (p <= 0,05). È attesa e poco informativa da sola.
- **P2 (centrale per la razionalità limitata).** L'angolo veloci–depth6 è maggiore dell'angolo classica–depth6, e il contrasto classica–veloci è significativo (p <= 0,05).
- **P3 (occupazione negli umani).** Nella classica il gap a materiale appaiato è positivo con placebo p <= 0,05 in almeno una prospettiva.
- **P4 (stesso obiettivo).** Nella classica e nei veloci le inversioni parziali rispetto a depth6 sono al massimo 1, come nel 3check e non come in antichess (3). Il materiale resta positivo con l'esito.
- **P5 (direzionale, bassa fiducia).** Il gap di occupazione a materiale appaiato è minore nei veloci che nella classica: con meno tempo, l'esito dipende meno dalla posizione occupata.

## Cosa conta come smentita
- **P2 falsa**: la cadenza non sposta la geometria in modo rilevabile, quindi il legame diretto con il limite di tempo cade.
- **P3 falsa**: l'occupazione resta un risultato sui motori.
- **P4 falsa**: il feature map non legge lo stesso obiettivo negli umani, e il trasferimento umani/motore va abbandonato.

## Confondenti dichiarati
- Aperture libere negli umani, fisse in depth6. La finestra primaria serve a mitigare questo punto.
- Abbandoni e patte d'accordo. Restano dentro e vanno dichiarati.
- Composizione degli esiti. Va riportata.
- Una parte dei tornei rapid/blitz è online. Va riportato quanti.
