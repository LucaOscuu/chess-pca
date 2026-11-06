# Predizioni registrate prima del test — "stessa mappa, flusso diverso" (04/10/2026, mattina)

Test esplorativo, scritto prima di guardare i numeri.

**Idea.** Ieri il piano PC1–PC2 è risultato uguale fra classica, rapid/blitz e Fishtest. Se la razionalità limitata lascia una traccia, non sta nella mappa ma in **come i giocatori si muovono sulla mappa**. Si guarda il campo di velocità medio (Friedman: cicli e campi di velocità nello spazio degli stati) e la direzione del movimento dopo uno shock (Selten, Neugebauer, Sadrieh: learning direction e impulse response).

**Dati.** Classica e rapid/blitz umani (da ieri), Fishtest (full) e depth6. Gli altri corpora non sono raggiungibili oggi, perché il PC non è collegato.

**Mappa comune.** PCA su un pool con lo stesso numero di partite per corpus. Le feature sono standardizzate sul pool e si usano le righe k >= 10 per prospettiva, cioè dopo l'apertura.

## Misure
- **M1 Velocità.** Lunghezza del passo |z_{t+1} − z_t| nel piano (PC1, PC2), a parità di stato materiale: in svantaggio (< −50 cp), pari, in vantaggio (> +50 cp).
- **M2 Campo di flusso.** Si costruisce una griglia 6×6 sui quantili del pool e, per ogni cella, la velocità media. La statistica è la distanza quadratica media fra i campi di due corpora, pesata per il numero di osservazioni. Il null permuta le partite fra i corpora (raggruppato per partita, 500 permutazioni).
- **M3 Risposta all'impulso.** L'impulso negativo è una perdita di materiale ≥ 100 cp per il giocatore fra la sua mossa t−1 e la t. La risposta è il passo successivo proiettato sulla direzione "vincente" u. La direzione u si stima su metà delle partite: è la differenza fra la posizione media di chi poi vince e di chi poi perde, a parità di materiale. La risposta si misura sull'altra metà.

## Predizioni
- **F1.** Classica e rapid/blitz hanno campi di flusso diversi (M2, p ≤ 0,05), anche se il piano è lo stesso.
- **F2.** Nei veloci il passo è più piccolo che nella classica quando si è in svantaggio (M1). È coerente con Carow & Witzig (JEBO 2025): sotto pressione di tempo si scelgono mosse più sicure, soprattutto in posizioni perdenti.
- **F3.** Dopo un impulso negativo, il passo successivo ha una proiezione positiva su u (learning direction: ci si muove verso la regione vincente). La risposta è più debole nei veloci.

**Smentite.**
- F1 falsa: la pressione di tempo non si vede nemmeno nella dinamica.
- F3 con proiezione ≤ 0: la direzione di movimento non contiene l'obiettivo.
