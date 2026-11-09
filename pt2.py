import sys; R='/tmp/claude-0/-home-claude/1bdb3634-9951-583d-9a91-beca2986667c/scratchpad/'
sys.path.insert(0,R+'pt_pt'); sys.path.insert(0,R+'lawsample_pkg/src')
from ptlib import *
import numpy as np
from lawsample.static_test import threshold_test
def mu0(X,S):
    A=np.c_[X[:,0]**2,np.ones(len(X))]; co=np.linalg.lstsq(A,X[:,1]**2,rcond=None)[0]
    r=threshold_test(X,"tw",S,law={"ww":1.0,"tt":-co[0],"1":-co[1]},degree=2,inject=(0.0,),m_grid=(10,),draws=2,reps=40,seed=1)[0]
    return r['mu0']
for name,L,seg in [('31.2cm.txt',31.2,[10,20,40,80]),('dados-quarto-esfera-61.5cm.txt',61.5,[8,16,32]),('dados-quarto-esfera-57.3cm.txt',57.3,[8,16,32])]:
    f=[v[1] for k,v in files.items() if k.endswith(name)][0]
    t,th,om=load(f); th=np.deg2rad(th)
    print(name)
    for T in seg:
        vals=[];loss=[]
        for t0 in np.arange(0,t[-1]-T,T):
            m=(t>=t0)&(t<t0+T)
            if m.sum()<60: continue
            X,S,sig=rows(t[m],th[m])
            if len(X)<12: continue
            E=X[:,0]**2+X[:,1]**2/ (9.8/(L/100)); 
            e1=E[:max(3,len(E)//5)].mean(); e2=E[-max(3,len(E)//5):].mean()
            vals.append(mu0(X,S)); loss.append(1-e2/e1)
        vals=np.array(vals)
        print(f"  segment {T:3d}s: {len(vals):2d} segs, rows/seg~{int(T/0.297)}, mu0 median {np.median(vals):.1f} (min {vals.min():.1f}, max {vals.max():.1f}); energy change over segment median {np.median(loss)*100:.1f}%")
