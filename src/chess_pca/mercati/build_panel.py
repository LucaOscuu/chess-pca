import pandas as pd, numpy as np, glob
P='work/mercati/rep/data/'
part=pd.read_stata(P+'proc/_participants.dta')
part['plab']=part.plab.astype(str)
rows=[]
for f in glob.glob(P+'raw/*/raw/cda_rep[12]_2*.csv'):
    d=pd.read_csv(f)
    d=d[d['subsession.practice']==0] if 'subsession.practice' in d else d
    rows.append(d[['participant.code','subsession.repetition','subsession.round_number','player.cash','player.assets','player.next_cash','group.dividend']].rename(columns={'participant.code':'plab','subsession.repetition':'rep','subsession.round_number':'period','player.cash':'cash','player.assets':'assets','player.next_cash':'next_cash','group.dividend':'div'}))
st=pd.concat(rows).dropna(subset=['cash']).drop_duplicates(['plab','rep','period'])
st=st.merge(part[['plab','slab','mid','pid','mlab','study','cond','traders']],on='plab',how='inner')
st['rep']=st.rep.astype(int); st['period']=st.period.astype(int)
o=pd.read_stata(P+'proc/orders.dta'); o['rep']=o.rep.cat.codes+1; o['slab']=o.slab.astype(str)
t=pd.read_stata(P+'proc/trades.dta'); t['rep']=t.rep.cat.codes+1
o['fv']=55-5*o.period
lim=o[o.order=='limit']
g=['slab','mid','pid','rep','period']
f=pd.DataFrame(index=o.groupby(g).size().index)
f['n_bid']=o[o.type=='bid'].groupby(g).size(); f['n_ask']=o[o.type=='ask'].groupby(g).size()
f['mkt_frac']=(o.order=='market').groupby([o[c] for c in g]).mean()
f['cancel']=(lim.deleted=='yes').groupby([lim[c] for c in g]).mean()
lb=lim[(lim.type=='bid')&(lim.price>0)]; la=lim[(lim.type=='ask')&(lim.price>0)]
f['bid_dev']=np.log(lb.price/lb.fv).groupby([lb[c] for c in g]).mean()
f['ask_dev']=np.log(la.price/la.fv).groupby([la[c] for c in g]).mean()
# own trades
oid=o.set_index('oid')[['slab','mid','pid','rep','period']]
tb=t.merge(oid,left_on='bid_id',right_index=True,suffixes=('','_o')); ta=t.merge(oid,left_on='ask_id',right_index=True,suffixes=('','_o'))
for side,tt in (('buy',tb),('sell',ta)):
    tt=tt.copy(); tt['fvv']=55-5*tt.period; tt['lp']=np.log(np.maximum(tt.price,1)/tt.fvv)*tt.volume
    gg=[tt['slab_o'],tt['mid_o'],tt['pid'],tt['rep_o'],tt['period_o']]
    q=tt.groupby(gg).volume.sum(); q.index.names=g; f[side+'_q']=q
    dv=tt.groupby(gg).lp.sum()/tt.groupby(gg).volume.sum(); dv.index.names=g; f[side+'_dev']=dv
f=f.reset_index()
st['slab']=st.slab.astype(str)
D=st.merge(f,on=g,how='left')
for c in ['n_bid','n_ask','buy_q','sell_q']: D[c]=D[c].fillna(0)
for c in ['mkt_frac','cancel','bid_dev','ask_dev','buy_dev','sell_dev']: D[c]=D[c].fillna(0)
D['fv_next']=np.maximum(55-5*(D.period+1),0)
D['wealth_fv']=D.next_cash+D.assets*D.fv_next
mk=['slab','mlab','rep','period']
for c,src in (('assets_rel','assets'),('cash_rel','next_cash'),('wealth_rel','wealth_fv')):
    D[c]=D[src]/D.groupby(mk)[src].transform('mean')
D['buy_q']=D.buy_q/D.groupby(mk).assets.transform('mean'); D['sell_q']=D.sell_q/D.groupby(mk).assets.transform('mean')
D['n_bid']=np.log1p(D.n_bid); D['n_ask']=np.log1p(D.n_ask)
e=pd.read_stata(P+'proc/earnings.dta'); e['rep']=e.rep.cat.codes+1; e['plab']=e.plab.astype(str)
D=D.merge(e[['plab','rep','earnings']],on=['plab','rep'],how='left')
D['med']=D.groupby(['slab','mlab','rep']).earnings.transform('median')
D['result_num']=np.sign(D.earnings-D.med).astype(int)
sk=pd.read_stata(P+'proc/skills.dta'); sk['plab']=sk.plab.astype(str)
D=D.merge(sk[['plab','skills','crt7']],on='plab',how='left')
D['traj']=D.plab+':'+D.rep.astype(str)
D=D.sort_values(['traj','period'])
D.to_parquet('work/mercati/work/panel.parquet',index=False)
print(D.shape, D.traj.nunique(), D.mlab.nunique()); print(D.groupby('rep').traj.nunique()); print(D.result_num.value_counts())
F=['assets_rel','cash_rel','wealth_rel','n_bid','n_ask','mkt_frac','cancel','bid_dev','ask_dev','buy_q','sell_q','buy_dev','sell_dev']
print(D[F].describe().T.round(3))
print(D.groupby('traj').size().value_counts().head())
