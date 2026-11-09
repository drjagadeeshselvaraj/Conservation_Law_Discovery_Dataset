import numpy as np, scipy.linalg as sl, sys, json
from scipy.signal import savgol_filter, savgol_coeffs
import sim_gen_lib as G
from trisim import feats, P
win=301; BL=1000; rg=np.random.default_rng(5); h0=savgol_coeffs(win,3,deriv=0)[win//2]
def smooth3(T,dt,idx): return ([savgol_filter(t,win,3)[idx] for t in T],[savgol_filter(t,win,3,deriv=1,delta=dt)[idx] for t in T],[savgol_filter(t,win,3,deriv=2,delta=dt)[idx] for t in T])
res={}
for c in (0.0,0.0005,0.002):
    mus=[]
    for i,a in enumerate([(1.2,0.9,0.5),(0.8,-1.1,1.4),(1.5,0.4,-0.9)],1):
        Y,t,dt=G.run(a,c,rg)
        T=[Y[k]+3e-5*rg.standard_normal(len(t)) for k in range(3)]
        L=len(t); idx=np.arange(win,L-win,200); n=len(idx)
        Tsm=[savgol_filter(x,win,3) for x in T]; RES=[x-s for x,s in zip(T,Tsm)]; Fb=feats(*smooth3(Tsm,dt,idx))[0]; N=0
        for _ in range(15):
            nb=L//BL+1; st=rg.integers(0,L-BL,nb); ix=np.concatenate([np.arange(s_,s_+BL) for s_ in st])[:L]
            Tp=[s+r[ix]/np.sqrt(1-h0) for s,r in zip(Tsm,RES)]; E=feats(*smooth3(Tp,dt,idx))[0]-Fb; N=N+E.T@E/n
        N/=15; F=feats(*smooth3(T,dt,idx))[0]; S=F.T@F/n; sc=np.sqrt(np.diag(S))
        mu=sl.eigh(S/np.outer(sc,sc),N/np.outer(sc,sc)+1e-12*np.eye(P),eigvals_only=True); mus.append(round(float(mu[0]),2))
    res[c]=mus; print("friction",c,"mu0 native per run",mus,flush=True)
json.dump(res,open("sim_mu0.json","w"))
