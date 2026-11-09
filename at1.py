import sys; sys.path.insert(0,'/tmp/claude-0/-home-claude/1bdb3634-9951-583d-9a91-beca2986667c/scratchpad/lawsample_pkg/src')
import numpy as np, pandas as pd
from lawsample.static_test import threshold_test
W=0.08
def load(f):
    d=pd.read_csv(f'data/{f}.csv',encoding='utf-8-sig'); d=d[d.ignore.isna()&d.dt.notna()].copy()
    d['key']=list(zip(d.angolo,d.flusso,d.massa)); return d
def sets():
    d1=load('sheet_1'); out={}
    for k,g in d1.groupby('key'): out['s1 %s'%(k,)]=g
    out['s3']=load('sheet_3'); return out
for name,g in sets().items():
    x=g.x_0.values/100.0; dt=g.dt.values; v=W/dt
    # relative timing noise from repeats (pooled)
    gr=g.groupby(['key','x_0']).dt; rel=np.sqrt(np.nanmean((gr.std()/gr.mean())**2))
    # physical check: v^2 linear in x ?
    A=np.c_[x,np.ones_like(x)]; co,res,_,_=np.linalg.lstsq(A,v**2,rcond=None)
    r=v**2-A@co; 
    X=np.c_[x,v]; sd=np.c_[np.zeros(len(x)),v*rel]
    t=threshold_test(X,"xv",sd,law=None,degree=2,inject=(0.0,),m_grid=(8,10,15,20,30,50),draws=60,reps=40,seed=3)[0]
    print('%-26s n=%3d rep-rel-noise %.4f  v^2=a x+b: a/2=%.3f (g sin th=%.3f) resid rel %.4f | mu0 %.2f mu1 %.1f A2 %.1f'%(name,len(x),rel,co[0]/2,9.80652*np.sin(np.radians(g.angolo.iloc[0])),r.std()/np.mean(v**2),t['mu0'],t['mu1'],t['A2']))
