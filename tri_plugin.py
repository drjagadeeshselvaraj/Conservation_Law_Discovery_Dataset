import numpy as np, scipy.linalg as sl, json
from scipy.signal import savgol_filter, savgol_coeffs
from tri import *
rg=np.random.default_rng(101); win=301; BL=1000; out={}
h0=savgol_coeffs(win,3,deriv=0)[win//2]
def smooth3(T,dt,idx):
    return ([savgol_filter(t,win,3)[idx] for t in T],[savgol_filter(t,win,3,deriv=1,delta=dt)[idx] for t in T],[savgol_filter(t,win,3,deriv=2,delta=dt)[idx] for t in T])
sigmas=[0.0003,0.001,0.003]; MS=(30,60,100,200,400,700)
def curve(rows_fn,Ntot,ref,Nc,reps=60):
    ca=lambda a:(a@Nc@ref)**2/((a@Nc@a)*(ref@Nc@ref)); res=[]
    for m in MS:
        al=[]
        for _ in range(reps):
            X=rows_fn(m); w,U=sl.eigh(X.T@X/m,Ntot); al.append(ca(U[:,0]))
        res.append(round(float(np.mean(al)),2))
    return res
for i in (1,2,3):
    T,_,dt=load(i); L=len(T[0]); idx=np.arange(win,L-win,200); n=len(idx)
    Tsm=[savgol_filter(t,win,3) for t in T]; RES=[t-sm for t,sm in zip(T,Tsm)]
    Fb=feats(*smooth3(Tsm,dt,idx))[0]
    Nnat=0
    for _ in range(15):
        nb=L//BL+1; st=rg.integers(0,L-BL,nb); ix=np.concatenate([np.arange(s_,s_+BL) for s_ in st])[:L]
        Tp=[s+r[ix]/np.sqrt(1-h0) for s,r in zip(Tsm,RES)]
        E=feats(*smooth3(Tp,dt,idx))[0]-Fb; Nnat=Nnat+E.T@E/n
    Nnat/=15
    Fc=feats(*smooth3(T,dt,idx))[0]; Sc=Fc.T@Fc/n; scc=np.sqrt(np.diag(Sc)); Fcs=Fc/scc
    Nc=Nnat/np.outer(scc,scc)+1e-12*np.eye(P); Scs=Sc/np.outer(scc,scc)
    muc,Uc=sl.eigh(Scs,Nc); refc=Uc[:,0]
    pool=np.arange(0,n,4)
    for sg in sigmas:
        Ninj=0
        for _ in range(10):
            Tp=[s+sg*rg.standard_normal(L) for s in Tsm]; E=feats(*smooth3(Tp,dt,idx))[0]-Fb; Ninj=Ninj+E.T@E/n
        Ninj/=10; Ntot=(Nnat+Ninj)/np.outer(scc,scc)+1e-12*np.eye(P); Lt=np.linalg.cholesky(Ntot)
        # (A) real clean rows + additive Gaussian feature noise N_tot  -> real signal structure, Gaussian noise
        cA=curve(lambda m:Fcs[rg.choice(pool,m,replace=False)]+rg.standard_normal((m,P))@Lt.T,Ntot,refc,Nc)
        # (B) Gaussian signal with the covariance of the clean rows + Gaussian noise -> the formula's assumptions
        Ls=np.linalg.cholesky(Scs+1e-10*np.eye(P))
        cB=curve(lambda m:rg.standard_normal((m,P))@Ls.T+rg.standard_normal((m,P))@Lt.T,Ntot,refc,Nc)
        print(f"run {i} sigma {sg}: A(real signal+Gauss noise) {cA} | B(Gauss signal+Gauss noise) {cB}",flush=True)
        out[f"{i}_{sg}"]=dict(A=cA,B=cB)
        json.dump(out,open("tri_plugin.json","w"))
