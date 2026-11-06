# Pre-registrazione — test sulla curva dei tassi (26/09/2026, 15:00, scritta prima di lanciare)

**Dati.** Rendimenti CMT dei Treasury USA, mensili, 1982-01 → 2012-12 (372 mesi), scadenze 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y. Fonte: pacchetto R `YieldCurve`, dataset `FedYieldCurve` (Federal Reserve H.15), dal mirror CRAN su GitHub. Recessioni NBER che iniziano dentro il campione: 1990-07, 2001-03, 2007-12.

**Traduzione del framework.** Sistema = la curva dei tassi USA. Traiettoria = il percorso mensile della curva. Strumento = scelte del ricercatore sulle stesse 8 scadenze. Geometria = piano delle prime due componenti (d = 8, loading confrontabili in R^8). Angolo fra piani = come per gli scacchi.

**Unità di ricampionamento.** Blocchi di 12 mesi consecutivi, l'equivalente delle partite. Ogni null permuta o divide blocchi, non mesi, per rispettare l'autocorrelazione.

**Strumenti** (stesse scadenze, scelte diverse del ricercatore):
- S1: livelli, PCA sulla correlazione (riferimento)
- S2: livelli, PCA sulla covarianza
- S3: variazioni mensili (primo differenziale), correlazione — la scelta di Litterman & Scheinkman
- S4: spread rispetto al 3M (y_m − y_3M, con il 3M stesso tenuto come livello), correlazione

**Periodi:** P1 = 1982-01 → 1994-12; P2 = 1995-01 → 2008-11; ZLB = 2008-12 → 2012-12 (49 mesi, tassi a breve fermi vicino allo zero).

## Y1 — Stabilità (A5)
Su 1982-01 → 2008-11: divisione casuale dei blocchi da 12 mesi in due metà, 500 volte, angolo fra i piani (S1). Si riportano mediana e p95. Descrittivo: forma dei loading (PC1 livello, stesso segno; PC2 pendenza, cambio di segno monotono fra corto e lungo).

## Y2 — Strumento contro sistema (C2)
- **Strumento:** angolo fra i piani S1-S2, S1-S3, S1-S4 sugli stessi mesi (1982-01 → 2008-11).
- **Sistema:** angolo fra P1 e P2 con S1 fisso, e fra (P1+P2) e ZLB con S1 fisso. Null: permutazione dei blocchi fra i due periodi (1.000 volte).
- **Ipotesi registrata (come negli scacchi):** l'angolo massimo dovuto allo strumento supera quello P1-P2.

## Y3 — Cambio di regole: lo ZLB come "antichess" dei tassi (A2)
Allo ZLB la parte corta della curva è bloccata dalla politica monetaria: le regole del sistema cambiano. **Previsioni registrate:**
1. la varianza di 3M e 6M allo ZLB scende di oltre il 90% rispetto a P1+P2;
2. la correlazione fra 3M e 10Y passa da > 0,8 (P1+P2) a < 0,5 (ZLB);
3. **controllo (come Three-Check):** l'angolo (P1+P2) contro ZLB supera il p95 del suo null, e supera l'angolo P1 contro P2, che è un cambio di periodo senza cambio di regole.

Y3 regge se valgono tutte e tre.

## Y4 — Occupazione e recessioni, attraverso gli strumenti
Regione definita senza esiti: quartile più basso del punteggio sulla componente di pendenza. Il segno è orientato in modo che valori alti corrispondano a una curva ripida, usando la correlazione con lo spread 10Y−3M, senza esiti. La regione si stima su tutto il campione per S1, S2 e S4. Esito: inizio di una recessione NBER nei 12 mesi successivi.

Statistica: quota di mesi nella regione nei 12 mesi prima di ciascun inizio di recessione, contro la quota negli altri mesi. **Con 3 eventi non c'è inferenza: è descrittivo.** Criterio di robustezza allo strumento: la quota pre-recessione supera la quota degli altri mesi per tutte e 3 le recessioni in tutti e tre gli strumenti di livello. S3 (variazioni) si riporta come esplorativo: le variazioni descrivono movimenti, non stati, quindi l'occupazione non è definita allo stesso modo.

## Limiti noti prima di lanciare
Una sola economia. 31 anni sono circa 31 blocchi indipendenti, quindi la potenza è bassa. Solo 3 recessioni. Lo ZLB ha 49 mesi. È un test di fattibilità del trasferimento, non la versione definitiva.
