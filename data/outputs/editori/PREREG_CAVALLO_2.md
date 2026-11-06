# Pre-registrazione — cavallo di Selten, secondo lancio (10/10/2026, ~21:40, prima del lancio)

**Perché.** Decisioni D1–D2 della scaletta della Sez. 3, approvate da Luca il 10/10:
- D1: la "zona efficiente" di `horse.py` (p3 > 0,8 fuori dall'imperfetto) non corrisponde a dove NSS (2025) osservano gli strangers nel lungo periodo;
- D2: `horse.py` stima una PCA separata per regola, mentre il registro parla di "PCA comune".

**Script.** `horse2.py` (stessa simulazione di `horse.py`, stesso seme 7, stesse regole e parametri; cambiano solo le misure e la mappa).

**Controllo di riproduzione (bloccante).** Le quote vicino all'equilibrio perfetto devono coincidere con `risultati_cavallo.json` (impulse 0,9646; reinforcement 0,2068) a 4 decimali. Se non coincidono, ci si ferma.

**Misure, sui periodi 150–249 come in `horse.py`.**
1. **Mappa comune:** PCA sulle osservazioni delle due regole messe insieme, feature (p1, p2, p3, u1, u2, u3) standardizzate insieme.
2. **Zona degli strangers di NSS** (da `nss2025_fatti.json`: p1 ≈ 0,5, p3 ≈ 0,7, p2 fra 0,3 e 0,4): |p1 − 0,5| ≤ 0,15, |p3 − 0,7| ≤ 0,15, 0,2 ≤ p2 ≤ 0,5. Quota di periodi nella zona, per regola.
3. **Zona dei partners** = equilibrio imperfetto, definizione invariata (p1 < 0,2, p2 > 0,3, p3 > 0,8).
4. **Sulla mappa:** proiezione del profilo degli strangers (0,5; 0,35; 0,7) con i payoff attesi di quel profilo, e del profilo imperfetto (0; 0,5; 1); per ciascuno, la quota del tempo osservato (per regola) in punti della mappa meno densi di quel punto (KDE gaussiana, come per RPS).

**Predizione.** Nessuna delle due regole passa più del 5% dei periodi nella zona degli strangers né nella zona dei partners, e i due profili umani cadono in zone poco dense della mappa comune (meno del 10% del tempo osservato in punti meno densi). Smentita se una delle due regole supera il 5% in una delle due zone.

**Cosa non cambia.** Regole, parametri, seme, disegno (60 gruppi da 9, strangers, 250 periodi, 100 giocate). I parametri restano scelti, non stimati.
