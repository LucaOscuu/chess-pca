# Predizioni — learning direction nella forma di Selten & Neugebauer (2019) (04/10/2026, 17:30)

**Origine.** Dopo il test C. Ho letto l'abstract di Selten & Neugebauer (2019, JEBO). La loro variabile di decisione sono gli **excess bids** (bid meno ask); la variabile di feedback è il **rendimento realizzato**, non il valore fondamentale. Il test C usava il fondamentale, quindi non era la loro teoria. Questo test è dichiarato come **successivo** al test C.

**Misure.**
- Variabile di decisione: eb_t = numero di bid − numero di ask del trader nel periodo t.
- Feedback: rendimento di mercato r_t = log(VWAP_t / VWAP_{t−1}).
- Learning direction: dopo r_t > 0 (ex post comprare di più sarebbe stato meglio) Δeb = eb_{t+1} − eb_t aumenta; dopo r_t < 0 diminuisce.

**Predizioni.**
- **D1.** La media di Δeb dopo rendimenti positivi è maggiore di quella dopo rendimenti negativi (p ≤ 0,05). Il test è per permutazione dei segni di r dentro ciascun mercato-ripetizione.
- **D2 (esplorativa).** La risposta D1 è più debole nei trader ad alta abilità cognitiva: ancorano al fondamentale, che scende, invece di seguire il rendimento.
