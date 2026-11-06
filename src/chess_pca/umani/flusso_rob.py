import re
src=open('flusso.py').read()
cut=src.index('# M1 velocità')
pre=src[:cut]
m2src=src[src.index('# M2 campo'):src.index('out["M2_campo"]')]
exec(pre)
import json
res={}
base=A.copy()
for lab,filt in [('solo_decisive',base.result_num!=0),('righe_10_40',base.k<=40),('decisive_e_10_40',(base.result_num!=0)&(base.k<=40))]:
    A=base[filt].copy()
    exec(m2src)
    res[lab]={f'{a} vs {b}':m2(a,b) for a,b in [('classica','veloci'),('classica','full'),('veloci','full')]}
print(json.dumps(res,indent=1))
json.dump(res,open('out/flusso_robustezza.json','w'),indent=1)
