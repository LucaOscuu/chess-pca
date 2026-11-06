# Controllo regressione verso la media: effetto dell'errore a parità di posizione corrente (bin 10x10 su p1,p2) e, per C1, di prezzo dei bid corrente
import numpy as np, pandas as pd, json
exec(open('bacino_ldt.py').read().split('out = {}')[0])
res={}
for lab,F in (("completo",FULL),("solo_comportamento",BEHAV)):
    m = RNG.permutation(markets); tr = set(m[: len(m) // 2])
    R, basin, sign, S = residences(F, tr)
    E = D[["traj","market","period","result_num","buy_q","sell_q"]].copy()
    E["bd"],E["sd"],E["bid"]=RAW_BUY_DEV.values,RAW_SELL_DEV.values,RAW_BIDDEV.values
    E["p1"],E["p2"]=S[:,0],S[:,1]; E["nb"]=D.n_bid.values
    itr=E.market.isin(tr)
    pos=E[itr&(E.result_num!=0)].groupby("traj").agg(p1=("p1","mean"),p2=("p2","mean"),r=("result_num","first"))
    u=(pos[pos.r==1][["p1","p2"]].mean()-pos[pos.r==-1][["p1","p2"]].mean()).to_numpy(); u=u/np.hypot(*u)
    E=E.sort_values(["traj","period"]); g=E.groupby("traj")
    E["proj"]=(g.p1.shift(-1)-E.p1)*u[0]+(g.p2.shift(-1)-E.p2)*u[1]
    E["dbid"]=g.bid.shift(-1)-E.bid; E["nbn"]=g.nb.shift(-1)
    E=E.dropna(subset=["proj"])
    E["err"]=(((E.buy_q>0)&(E.bd>0.1))|((E.sell_q>0)&(E.sd<-0.1))).astype(float)
    E["over"]=((E.buy_q>0)&(E.bd>0.1)).astype(float)
    te=E[~E.market.isin(tr)].copy()
    te["cell"]=pd.qcut(te.p1,10,labels=False).astype(str)+"_"+pd.qcut(te.p2,10,labels=False).astype(str)+"_"+te.period.astype(str)
    # effetto dentro cella (fixed effects): demean per cella
    for y,x,dd in (("proj","err",te),("dbid","over",te[(te.nb>0)&(te.nbn>0)].copy())):
        if y=="dbid": dd["cell"]=pd.qcut(dd.bid.rank(method='first'),10,labels=False).astype(str)+"_"+dd.period.astype(str)
        yd=dd[y]-dd.groupby("cell")[y].transform("mean"); xd=dd[x]-dd.groupby("cell")[x].transform("mean")
        beta=float((xd*yd).sum()/(xd*xd).sum())
        # bootstrap per mercato
        mk=dd.market.to_numpy(); um=np.unique(mk); idx={k:np.where(mk==k)[0] for k in um}; bs=[]
        yv,xv=yd.to_numpy(),xd.to_numpy()
        for _ in range(1000):
            ii=np.concatenate([idx[k] for k in RNG.choice(um,len(um))]); bs.append((xv[ii]*yv[ii]).sum()/(xv[ii]**2).sum())
        res[f"{lab}|{y}~{x}"]=dict(beta_entro_cella=round(beta,4),ic95=[round(float(np.percentile(bs,2.5)),4),round(float(np.percentile(bs,97.5)),4)],n=int(len(dd)))
print(json.dumps(res,indent=1)); json.dump(res,open('risultati_ldt_controllo.json','w'),indent=1)
