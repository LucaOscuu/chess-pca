import numpy as np, pandas as pd, json
P='work/mercati/rep/data/proc/'
D=pd.read_parquet('panel.parquet'); D['market']=D.slab+':'+D.mlab
t=pd.read_stata(P+'trades.dta'); t['rep']=t.rep.cat.codes+1; t['slab']=t.slab.astype(str)
v=t.groupby(['slab','mid','rep','period']).apply(lambda x:(x.price*x.volume).sum()/x.volume.sum()).rename('vwap').reset_index()
D=D.merge(v,on=['slab','mid','rep','period'],how='left')
D=D.sort_values(['traj','period'])
D['eb']=np.expm1(D.n_bid)-np.expm1(D.n_ask)
g=D.groupby('traj')
D['r']=np.log(D.vwap/g.vwap.shift(1))
D['deb']=g.eb.shift(-1)-D.eb
D['skill_hi']=D.skills>D.groupby('market').skills.transform('median')
E=D.dropna(subset=['r','deb'])
E=E[E.r!=0]; E['pos']=(E.r>0)
mr=(E.market+':'+E.rep.astype(str)).to_numpy()
rng=np.random.default_rng(20261004)
def stat(e,pos): return e.deb[pos].mean()-e.deb[~pos].mean()
def test(e):
    obs=stat(e,e.pos.to_numpy())
    # permuta il segno del rendimento a livello mercato-ripetizione-periodo dentro mercato-ripetizione
    key=(e.market+':'+e.rep.astype(str)).to_numpy(); per=e.period.to_numpy(); p0=e.pos.to_numpy()
    cells=pd.DataFrame({'k':key,'per':per,'p':p0}).drop_duplicates(['k','per'])
    null=[]
    for _ in range(2000):
        cp=cells.copy(); cp['p']=cp.groupby('k').p.transform(lambda s: rng.permutation(s.to_numpy()))
        m=pd.DataFrame({'k':key,'per':per}).merge(cp,on=['k','per'],how='left').p.to_numpy().astype(bool)
        null.append(stat(e,m))
    null=np.array(null)
    return dict(diff=round(float(obs),3),deb_dopo_pos=round(float(e.deb[e.pos].mean()),3),deb_dopo_neg=round(float(e.deb[~e.pos].mean()),3),
                p=round(float((np.sum(np.abs(null)>=abs(obs))+1)/2001),4),n=int(len(e)))
out={'D1_tutti':test(E),'D2_alta_abilita':test(E[E.skill_hi]),'D2_bassa_abilita':test(E[~E.skill_hi])}
# interazione: differenza delle differenze, permutando l'etichetta di abilità fra trader dentro mercato
hi=out['D2_alta_abilita']['diff']; lo=out['D2_bassa_abilita']['diff']; obs=hi-lo
lab=E.groupby('traj').skill_hi.first(); mk=E.groupby('traj').market.first(); null=[]
for _ in range(1000):
    pl=lab.groupby(mk).transform(lambda s: rng.permutation(s.to_numpy())); m=E.traj.map(pl).to_numpy().astype(bool)
    null.append(stat(E[m],E.pos.to_numpy()[m])-stat(E[~m],E.pos.to_numpy()[~m]))
out['D2_alta_meno_bassa']=dict(diff=round(float(obs),3),p=round(float((np.sum(np.abs(null)>=abs(obs))+1)/1001),4))
print(json.dumps(out,indent=1)); json.dump(out,open('risultati_ldt2.json','w'),indent=1)
