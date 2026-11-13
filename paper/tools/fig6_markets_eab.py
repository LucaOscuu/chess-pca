"""Figura nuova (Sez. 3.3): dove vanno i trader vincenti e cosa c'e' nella regione (EAB) dei mercati.
A: quota di periodi nell'EAB per periodo, vincenti vs perdenti, meta di test, ricchezza appaiata (0.9-1.1).
B: cosa distingue le osservazioni dentro l'EAB da quelle fuori (differenza in deviazioni standard), stessa selezione.
C: learning direction: stima grezza vs stima a parita' di posizione e periodo, in deviazioni standard dell'esito.
Legge: panel.parquet (build_panel.py), occ_maps.pkl (copia di analisi_mercati.py che salva la mappa della meta di
training, stesso seed), risultati_ldt_grezzo_vs_cella.json (stessa procedura di ldt_controllo.py)."""
import pickle, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W='work/mercati/work/'
S=pickle.load(open(W+'occ_maps.pkl','rb'))[('completo',True)]
D=pd.read_parquet(W+'panel.parquet'); RAW=D.copy()
FULL=S['F']
for c in FULL:
    lo,hi=D[c].quantile([.01,.99]); D[c]=D[c].clip(lo,hi)
D['market']=D.slab+':'+D.mlab
P=((D[FULL].to_numpy(float)-S['mu'])/S['sd']-S['zm'])@S['Wt']; q=S['q']; j=S['j']
inreg=[P[:,0]<q[0,0],P[:,0]>q[1,0],P[:,1]<q[0,1],P[:,1]>q[1,1]][j]
eab=inreg if S['s']>0 else ~inreg
test=~D.market.isin(set(S['tr'])).to_numpy(); keep=D.wealth_rel.between(0.9,1.1).to_numpy()
sel=test&keep
E=D.loc[sel,['market','traj','period','result_num']].copy(); E['eab']=eab[sel].astype(float)
E=E[E.result_num!=0]
RNG=np.random.default_rng(20261107)
# A: per periodo
per=E.groupby(['result_num','period']).eab.mean().unstack(0)
mk=E.market.unique(); g={m:E[E.market==m] for m in mk}; boots=[]
for _ in range(1000):
    Eb=pd.concat([g[m] for m in RNG.choice(mk,len(mk))]); boots.append(Eb.groupby(['result_num','period']).eab.mean().unstack(0))
lo=pd.concat(boots).groupby(level=0).quantile(.025); hi=pd.concat(boots).groupby(level=0).quantile(.975)
# check: gap a livello di traiettoria come in analisi (min 3 periodi)
T=E.groupby('traj').agg(eab=('eab','mean'),n=('eab','size'),r=('result_num','first')); T=T[T.n>=3]
gap=100*(T[T.r==1].eab.mean()-T[T.r==-1].eab.mean())
print('gap traiettorie (deve essere 7.54):',round(gap,2))
# B: profilo
feats=[('buy_q','Units bought'),('buy_dev','Purchase price vs value'),('mkt_frac','Share of market orders'),('n_bid','Bids placed'),
       ('assets_rel','Asset holdings'),('cash_rel','Cash'),('ask_dev','Ask price vs value'),('sell_q','Units sold')]
R=RAW.loc[sel]; e=eab[sel]
prof=[(lab,(R.loc[e,c].mean()-R.loc[~e,c].mean())/R[c].std(), R.loc[e,c].mean(), R.loc[~e,c].mean()) for c,lab in feats]
# C: LDT
L=json.load(open(W+'risultati_ldt_grezzo_vs_cella.json'))
fig,ax=plt.subplots(1,3,figsize=(11.2,3.5),gridspec_kw=dict(width_ratios=[1.05,1.1,1.05],wspace=0.55))
BLUE,ORANGE,INK,MUTED="#2a78d6","#eb6834","#1f1f1e","#8a8a85"
a=ax[0]; t=per.index.to_numpy()
for r,col,labl in ((1,BLUE,'Winners'),(-1,ORANGE,'Losers')):
    a.fill_between(t,100*lo[r],100*hi[r],color=col,alpha=.18,lw=0); a.plot(t,100*per[r],'-o',color=col,ms=3.5,lw=1.6,label=labl)
a.set_xlabel('Period'); a.set_ylabel('Share in the EAB (%)'); a.set_xticks(range(1,11)); a.legend(frameon=False,fontsize=8)
a.set_title('A. Time in the EAB, by period',fontsize=9.5,loc='left')
b=ax[1]; labs=[p[0] for p in prof][::-1]; vals=[p[1] for p in prof][::-1]
b.barh(range(len(vals)),vals,color=["#3d3d3a" if v<0 else "#b9b8b1" for v in vals],height=.62)
b.set_yticks(range(len(vals))); b.set_yticklabels(labs,fontsize=8); b.axvline(0,color=INK,lw=.8)
b.set_xlabel('Inside minus outside the EAB (s.d.)'); b.set_title('B. What the EAB contains',fontsize=9.5,loc='left')
c=ax[2]; rows=[('Step toward winners\nafter a mistake','completo|proj~err'),('Bid change\nafter overpaying','completo|dbid~over')]
for i,(lab,k) in enumerate(rows):
    v=L[k]; sd=v['sd_y']
    for off,(est,ci,col,mk_) in zip((-.14,.14),((v['beta_grezzo'],v['ic95_grezzo'],MUTED,'s'),(v['beta_entro_cella'],v['ic95'],INK,'o'))):
        c.errorbar(est/sd,i+off,xerr=[[ (est-ci[0])/sd],[(ci[1]-est)/sd]],fmt=mk_,color=col,ms=5,capsize=3,lw=1.3)
c.axvline(0,color=INK,lw=.8); c.set_yticks([0,1]); c.set_yticklabels([r[0] for r in rows],fontsize=8); c.set_ylim(-.6,1.6); c.invert_yaxis()
c.set_xlabel('Effect (s.d. of the outcome)'); c.set_title('C. Learning direction',fontsize=9.5,loc='left')
c.plot([],[],'s',color=MUTED,label='Raw'); c.plot([],[],'o',color=INK,label='Same position and period'); c.legend(frameon=False,fontsize=7.5,loc='lower right')
for x in ax:
    for s_ in ('top','right'): x.spines[s_].set_visible(False)
fig.savefig('fig6_markets_eab.pdf',bbox_inches='tight'); fig.savefig('fig6_markets_eab.png',dpi=200,bbox_inches='tight')
out=dict(gap_check=round(gap,2),per_period={int(k):{'win':round(100*per.loc[k,1],1),'lose':round(100*per.loc[k,-1],1)} for k in per.index},
         profile={p[0]:dict(sd_diff=round(p[1],2),inside=round(p[2],3),outside=round(p[3],3)) for p in prof},
         ldt={k:dict(raw_sd=round(L[k]['beta_grezzo']/L[k]['sd_y'],3),cell_sd=round(L[k]['beta_entro_cella']/L[k]['sd_y'],3)) for _,k in rows})
json.dump(out,open(W+'fig6_numeri.json','w'),indent=1); print(json.dumps(out,indent=1))
