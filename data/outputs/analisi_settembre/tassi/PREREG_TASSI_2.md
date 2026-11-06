# Pre-registrazione — test rafforzati sui tassi (27/09/2026, 12:10, scritta prima di lanciare)

**Dati.**
- **CMT**: Treasury USA, rendimenti par a scadenza costante, mensili 1982-01 → 2012-12, 8 scadenze (3M–10Y). `YieldCurve::FedYieldCurve`.
- **MK**: McCulloch & Kwon (1993), rendimenti zero-coupon stimati con spline, mensili 1946-12 → 1991-02, 10 scadenze (1M–120M). `Ecdat::Irates`.
- **ECB**: area euro, rendimenti spot AAA, giornalieri lavorativi 2006-12-29 → 2009-07-24, 32 scadenze (3M–30Y). `YieldCurve::ECBYieldCurve`.

CMT e MK sono **due strumenti indipendenti sullo stesso sistema** (fonte, metodo di costruzione e scadenze diverse), sovrapposti fra 1982-01 e 1991-02 (110 mesi).

**Recessioni NBER (mese di picco)** usate come esiti: 1948-11, 1953-07, 1957-08, 1960-04, 1969-12, 1973-11, 1980-01, 1981-07 (dataset MK); 1990-07, 2001-03, 2007-12 (dataset CMT). Totale 11 eventi, ciascuno assegnato a un solo dataset.

**Unità di ricampionamento:** blocchi di 12 mesi (MK, CMT) o di 20 giorni lavorativi (ECB).

## R1 — Occupazione prima delle recessioni, con inferenza (11 eventi)
Per ciascun dataset: PCA dei livelli sulla correlazione. La componente di pendenza è quella delle prime due più correlata con lo spread lungo−corto (MK: r120−r1; CMT: 10Y−3M), orientata in modo che valori alti significhino curva ripida. Regione = quartile più basso, definito su tutto il dataset senza esiti.
Statistica: media, sugli 11 eventi, della quota di mesi nella regione nei 12 mesi prima del picco, meno la quota nei mesi fuori da tutte le finestre pre-recessione.
**Null:** le date delle recessioni di ciascun dataset vengono traslate insieme di uno stesso intervallo circolare casuale (5.000 volte). Questo conserva la spaziatura fra gli eventi e l'autocorrelazione della serie.
**Criterio.** Regge se p < 0,05 e se almeno 9 recessioni su 11 hanno quota pre-recessione superiore a quella degli altri mesi.
**Robustezza allo strumento:** stesso test con PCA sulla covarianza e con MK ristretto alle scadenze più vicine a CMT (3, 6, 12, 36, 60, 120 mesi). Solo riportato.
**Fuori campione nel tempo:** soglia della regione stimata solo su 1946–1969 (MK), applicata agli eventi 1969–1981. Solo riportato.

## R2 — Convergenza fra due strumenti indipendenti (A6 sui tassi)
Periodo comune 1982-01 → 1991-02. Livelli standardizzati.
- Angolo fra i percorsi mensili recuperati dalle PCA dei due strumenti (piani delle prime due componenti nello spazio delle osservazioni).
- Correlazioni canoniche fra i due insiemi di rendimenti: prime due.
- Accordo sulla regione "curva piatta" (quartile più basso della pendenza, definito separatamente in ciascuno strumento): kappa di Cohen.
**Criterio.** A6 regge se le prime due correlazioni canoniche sono > 0,9 e il kappa è > 0,6.

## R3 — L'effetto dello strumento supera il rumore di campionamento
Per MK (1946–1991) e CMT (1982–2008-11): angolo fra i piani con livelli su correlazione (S1) e con variazioni mensili (S3). Intervallo al 95% con bootstrap a blocchi di 12 mesi (500 volte), confrontato con il p95 del null split-half a blocchi dello stesso dataset.
**Criterio.** Regge se in entrambi i dataset il limite inferiore dell'intervallo supera il p95 del null.

## R4 — Scomposizione della variabilità (PERMANOVA) su 45 anni di MK
Sistemi = tre periodi monetari: 1947–1965 (Bretton Woods, tassi bassi), 1966–1979 (inflazione), 1979-10 → 1991-02 (Volcker e dopo). Strumenti = S1 (livelli, correlazione), S2 (livelli, covarianza), S3 (variazioni, correlazione). 5 metà casuali a blocchi di 12 mesi per ogni cella. Distanza di proiezione fra piani (d = 10).
**Ipotesi registrata:** quota strumento > quota sistema, come negli scacchi (51% contro 32%).

## R5 — Seconda economia: area euro (ECB)
- A5: split-half a blocchi di 20 giorni, p95.
- C2: angolo livelli (correlazione) contro variazioni giornaliere, con intervallo bootstrap a blocchi.
- Sistema: prima contro dopo il fallimento di Lehman (2008-09-15). La data è approssimata contando i giorni lavorativi dal 2006-12-29 (errore di qualche giorno per le festività); null a permutazione di blocchi.
**Criterio.** Replica C2 se il limite inferiore dell'intervallo livelli-variazioni supera il p95 del null split-half. Il confronto con l'angolo pre/post Lehman è solo riportato.
