import numpy as np, scipy.linalg as sl, json, sys
from scipy.signal import savgol_filter
from tri import *
rg=np.random.default_rng(81); win=201; BL=1000; out={}
h0=savgol_coeffs=None
from scipy.signal import savgol_coeffs
h0=savgol_coeffs(win,3,deriv=0)[win//2]
def smooth3(T,dt,idx):
    return ([savgol_filter(t,win,3)[idx] for t in T],[savgol_filter(t,win,3,deriv=1,delta=dt)[idx] for t in T],[savgol_filter(t,win,3,deriv=2,delta=dt)[idx] for t in T])
sigmas=[0.0,0.0003,0.001,0.003]
for i in (1,2,3):
    T,_,dt=load(i); L=len(T[0]); idx=np.arange(win,L-win,200); n=len(idx)
    Tsm=[savgol_filter(t,win,3) for t in T]; RES=[t-sm for t,sm in zip(T,Tsm)]
    Fb=feats(*smooth3(Tsm,dt,idx))[0]
    # native noise by block bootstrap of residuals
    Nnat=0; reps=15
    for _ in range(reps):
        nb=L//BL+1; st=rg.integers(0,L-BL,nb); ix=np.concatenate([np.arange(s_,s_+BL) for s_ in st])[:L]
        Tp=[s+r[ix]/np.sqrt(1-h0) for s,r in zip(Tsm,RES)]
        E=feats(*smooth3(Tp,dt,idx))[0]-Fb; Nnat=Nnat+E.T@E/n
    Nnat/=reps
    F0c=feats(*smooth3(T,dt,idx))[0]; Sc=F0c.T@F0c/n; scc=np.sqrt(np.diag(Sc))
    Nc=Nnat/np.outer(scc,scc)+1e-12*np.eye(P); Scs=Sc/np.outer(scc,scc)
    muc,Uc=sl.eigh(Scs,Nc); refc=Uc[:,0]                   # clean reference direction (standardised coords)
    for sg in sigmas:
        Tn=[t+sg*rg.standard_normal(L) for t in T]; Fn=feats(*smooth3(Tn,dt,idx))[0]
        Ninj=0
        if sg>0:
            for _ in range(10):
                Tp=[s+sg*rg.standard_normal(L) for s in Tsm]
                E=feats(*smooth3(Tp,dt,idx))[0]-Fb; Ninj=Ninj+E.T@E/n
            Ninj/=10
        Ntot=(Nnat+Ninj)/np.outer(scc,scc)+1e-12*np.eye(P)
        S=Fn.T@Fn/n; Ss=S/np.outer(scc,scc); Fs=Fn/scc
        mu,U=sl.eigh(Ss,Ntot); lam=mu[1:]-1; nneg=int((lam<=0).sum()); lam=lam[lam>0]
        A1=np.sum(1+1/lam); A2=np.sum((1+1/lam)**2)
        ca=lambda a:(a@Nc@refc)**2/((a@Nc@a)*(refc@Nc@refc))
        pool=np.arange(0,n,4); rows=[]
        for m in (30,60,100,200,400,700):
            if m>len(pool): continue
            al=[]
            for _ in range(80):
                ii=rg.choice(pool,m,replace=False); Sn=Fs[ii].T@Fs[ii]/m; w,Un=sl.eigh(Sn,Ntot); al.append(ca(Un[:,0]))
            rows.append((m,round(float(np.mean(al)),2),round(max(0,(1-A2/m)/(1-A1/m)),2) if m>A2 else 0.0))
        gap=float(np.sqrt(np.mean([(a-b)**2 for _,a,b in rows])))
        # measured transition: first m with mean alignment >= 0.5
        mt=next((m for m,a,_ in rows if a>=0.5),None)
        print(f"run {i} sigma_inj {sg}: mu0 {mu[0]:.2f} A1 {A1:.0f} A2 {A2:.0f} (neg {nneg}) rms gap {gap:.2f} meas50 {mt} {rows}",flush=True)
        out[f"{i}_{sg}"]=dict(A1=float(A1),A2=float(A2),gap=gap,meas50=mt,rows=rows,mu0=float(mu[0]))
        json.dump(out,open("tri_inject_w201.json","w"))
