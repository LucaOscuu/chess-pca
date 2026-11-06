# Predizioni registrate prima del test — poker (04/10/2026, 17:00)

**Dati.** Dataset HandHQ (uoftcprg/phh-dataset): mani di No-Limit Hold'em online su PokerStars, luglio 2009, con giocatori anonimizzati ma identificativo persistente.
- 40 file a **25NL** (bassa posta, buio grande $0,25), circa 40.000 mani.
- 40 file a **1000NL** (alta posta, buio grande $10), circa 40.000 mani.

**Perché il poker.** Completa la terza fonte di razionalità limitata: incertezza, caso e informazione nascosta. Le regole sono note, la strategia ottimale non è calcolabile.

**Unità.** Un giocatore con almeno 200 mani a una posta.

**Traiettoria.** Blocchi consecutivi di 10 mani (massimo 30 blocchi, cioè 300 mani) in ordine di tempo.

**Feature per blocco (solo comportamento, niente denaro).** Quota di mani in cui il giocatore:
- mette soldi volontariamente preflop (VPIP);
- rilancia preflop (PFR);
- folda preflop;
- vede il flop;
- arriva allo showdown.

Più tre misure di aggressività:
- rilanci per azione preflop;
- rilanci per azione postflop;
- fattore di aggressione (bet + raise)/(call + 1).

**Esito.** Il giocatore vince se il risultato netto su tutte le sue mani del campione è > 0 (vincite registrate meno importo investito, con restituzione delle puntate non chiamate).

## Test
- **T0.** Lo spettro supera il null a colonne mescolate e il piano è stabile su metà dei giocatori (angolo split-half mediano < 30°).
- **T1.** Mappa bassa vs alta posta. **Predizione:** l'angolo supera il null (p ≤ 0,05). La posta cambia la popolazione dei giocatori, come abilità ed esperienza nei mercati.
- **T2 (bacino).** Regioni dai quartili di PC1 e PC2 su metà dei giocatori. Il bacino è la regione con il gap vincitori − perdenti più grande sul training; si misura sull'altra metà. Il placebo permuta gli esiti fra giocatori della stessa posta (1.000 permutazioni). **Predizione:** gap > 0 con p ≤ 0,05, separatamente a ciascuna posta.
- **T3 (flusso).** Il campo di velocità differisce fra bassa e alta posta (p ≤ 0,05).
- **Validità di contenuto (descrittiva).** Il bacino corrisponde allo stile "tight-aggressive" (VPIP bassa, aggressività alta), che la conoscenza del poker associa alla vittoria. Il metodo ritroverebbe dai soli dati una regola nota agli esperti, senza che gli sia stata data.

## Cautela dichiarata prima
Nel poker la varianza dovuta al caso è enorme: con 200–300 mani, l'esito "vince/perde" è in larga parte fortuna. Un T2 nullo sarebbe quindi poco informativo; un T2 positivo è notevole.

## Aggiunta prima del secondo lancio (17:20)

**Primo lancio.** T2 è risultato nullo a entrambe le poste (AUC 0,50 e 0,46), come avvertito: con 200–300 mani l'esito è dominato dal caso.

**Robustezza dichiarata prima di rilanciare.** Si scaricano 200 file per posta invece di 40. Si alza la soglia a giocatori con almeno 1.000 mani. L'esito si calcola su tutte le mani del giocatore; la traiettoria resta sulle prime 300 mani, in blocchi da 10.

**Predizione** invariata: gap > 0 con p ≤ 0,05. Se resta nullo, il poker entra nel paper come caso in cui il bacino non si recupera con questi dati.
