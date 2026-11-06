import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
exec(open('bacino_ldt.py').read().split('def test_B')[0])
out={}
for lab,F in (("completo",FULL),("solo_comportamento",BEHAV)):
    rows=[]
    for s in range(20):
        m=RNG.permutation(markets); tr=set(m[:len(m)//2])
        R,basin,_,_=residences(F,tr)
        A=D.groupby('traj')[F].mean(); A=A.join(R[['res','train','basin']],how='inner'); A=A[A.res!=0]
        trn,tst=A[A.train],A[~A.train]; ytr=(trn.res==1).astype(int); yte=(tst.res==1).astype(int)
        auc_basin=roc_auc_score(yte,tst.basin)
        mu,sd=trn[F].mean(),trn[F].std()
        lr=LogisticRegression(max_iter=2000).fit((trn[F]-mu)/sd,ytr); auc_lr=roc_auc_score(yte,lr.predict_proba((tst[F]-mu)/sd)[:,1])
        km=KMeans(4,n_init=10,random_state=s).fit((trn[F]-mu)/sd); ctr=km.predict((trn[F]-mu)/sd); cte=km.predict((tst[F]-mu)/sd)
        rate=pd.Series(ytr.values).groupby(ctr).mean(); auc_km=roc_auc_score(yte,pd.Series(cte).map(rate).fillna(.5))
        rows.append(dict(basin=auc_basin,logit=auc_lr,kmeans=auc_km))
    X=pd.DataFrame(rows); out[lab]={c:[round(float(X[c].median()),3),round(float(X[c].quantile(.1)),3),round(float(X[c].quantile(.9)),3)] for c in X}
print(json.dumps(out,indent=1)); json.dump(out,open('risultati_baseline.json','w'),indent=1)
