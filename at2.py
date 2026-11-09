# Air-track incline (neneone/airtrack-data-analysis, real data): threshold test, v^2 = alpha x + beta (energy conservation)
import sys; sys.path.insert(0,'/tmp/claude-0/-home-claude/1bdb3634-9951-583d-9a91-beca2986667c/scratchpad/lawsample_pkg/src')
import numpy as np, pandas as pd, json
from lawsample.static_test import threshold_test
W=0.08
d=pd.read_csv('data/sheet_1.csv',encoding='utf-8-sig'); d=d[d.ignore.isna()&d.dt.notna()].copy()
out={}
for k,g in d.groupby(['angolo','flusso','massa']):
    x=g.x_0.values/100.0; v=W/g.dt.values
    gr=g.groupby('x_0').dt; rel=float(np.sqrt(np.mean((gr.std()/gr.mean())**2)))
    A=np.c_[x,np.ones_like(x)]; co=np.linalg.lstsq(A,v**2,rcond=None)[0]
    law={"vv":1.0,"x":-co[0],"1":-co[1]}
    X=np.c_[x,v]; sd=np.c_[np.zeros(len(x)),v*rel]
    res=threshold_test(X,"xv",sd,law=law,degree=2,inject=(0.0,0.003,0.01,0.02),inject_cols=[1],inject_mode="std",
                       m_grid=(6,8,10,12,15,20,25,30),draws=80,reps=40,seed=5)
    print(k,'n',len(x),'rel noise %.4f'%rel)
    for r in res:
        print('  inject %.3f*std(v): mu0 %.2f A2 %.1f n05 %s rms %.3f'%(r['inject'],r['mu0'],r['A2'],r['n05'],r['rms_gap']),
              ' | '.join('%d:%.2f/%.2f'%(m,a,f) for m,a,f,p in r['rows']))
    out[str(k)]=dict(n=len(x),rel=rel,res=res)
json.dump(out,open('at2.json','w'),default=float)
