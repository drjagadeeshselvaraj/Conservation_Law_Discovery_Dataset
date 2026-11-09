import numpy as np, scipy.linalg as sl, json, sys
from scipy.signal import savgol_filter
from tri import *
from nnk import project_to_library
from tuned import sindy_tuned, nn_tuned, kernel_tuned
import time
rg=np.random.default_rng(81); win=201; BL=1000; out={}
h0=savgol_coeffs=None
from scipy.signal import savgol_coeffs
h0=savgol_coeffs(win,3,deriv=0)[win//2]
def smooth3(T,dt,idx):
    return ([savgol_filter(t,win,3)[idx] for t in T],[savgol_filter(t,win,3,deriv=1,delta=dt)[idx] for t in T],[savgol_filter(t,win,3,deriv=2,delta=dt)[idx] for t in T])

def stlsq_null(Fm,tau=0.1,iters=6):
    best=None
    P_=Fm.shape[1]
    for j in range(P_):
        idx=[q for q in range(P_) if q!=j]
        A=Fm[:,idx]; b=-Fm[:,j]
        c=np.linalg.lstsq(A,b,rcond=None)[0]
        for _ in range(iters):
            sm=np.abs(c)<tau; c[sm]=0
            big=~sm
            if big.sum()==0: break
            c[big]=np.linalg.lstsq(A[:,big],b,rcond=None)[0]
        v=np.zeros(P_); v[idx]=c; v[j]=1.0
        r=np.linalg.norm(Fm@v)/np.sqrt(len(Fm))/np.linalg.norm(v)
        if best is None or r<best[0]: best=(r,v/np.linalg.norm(v))
    return best[1]
sigmas=[0.0003,0.001]
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
        Tn=[t+sg*rg.standard_normal(L) for t in T]; S3=smooth3(Tn,dt,idx); Fn,Phin=feats(*S3); Xs=np.column_stack(S3[0]+S3[1]); Xds=np.column_stack(S3[1]+S3[2]); xm=Xs.mean(0); xs_=Xs.std(0); Xs=(Xs-xm)/xs_; Xds=Xds/xs_
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
        for m in (60,100,200,400):
            if m>len(pool): continue
            an=[];ak=[];aw=[];asd=[];cfg=[]
            for _ in range(6):
                ii=rg.choice(pool,m,replace=False)
                Sn=Fs[ii].T@Fs[ii]/m; w,Un=sl.eigh(Sn,Ntot); aw.append(ca(Un[:,0]))
                Qn,H=nn_tuned(Xs[ii],Xds[ii],rg); c=project_to_library(Qn(Xs[pool]),Phin[pool]); an.append(ca(c*scc))
                Qk,hp=kernel_tuned(Xs[ii],Xds[ii],rg); c=project_to_library(Qk(Xs[pool]),Phin[pool]); ak.append(ca(c*scc))
                v,sc=sindy_tuned(Fs[ii],rg=rg); asd.append(ca(v)); cfg.append((H,hp,sc))
            rows.append((m,round(float(np.mean(aw)),2),0.0,round(float(np.mean(an)),2),round(float(np.mean(ak)),2),round(float(np.mean(asd)),2)))
            print(' ',m,rows[-1],flush=True)
        gap=float(np.sqrt(np.mean([(a-b)**2 for _,a,b,_c,_d,_e in rows])))
        # measured transition: first m with mean alignment >= 0.5
        mt=next((m for m,a,_,_c,_d,_e in rows if a>=0.5),None)
        print(f"run {i} sigma_inj {sg}: mu0 {mu[0]:.2f} A1 {A1:.0f} A2 {A2:.0f}",flush=True)
        out[f"{i}_{sg}"]=dict(A1=float(A1),A2=float(A2),gap=gap,meas50=mt,rows=rows,mu0=float(mu[0]))
        json.dump(out,open("tri_tuned.json","w"))
