import numpy as np, scipy.linalg as sl, json, pickle
from scipy.signal import savgol_filter, savgol_coeffs
from tri import *
rg=np.random.default_rng(121); win=301; BL=1000; MS=(30,60,100,200,400,700)
h0c=savgol_coeffs(win,3,deriv=0)[win//2]
def smooth3(T,dt,idx):
    return ([savgol_filter(t,win,3)[idx] for t in T],[savgol_filter(t,win,3,deriv=1,delta=dt)[idx] for t in T],[savgol_filter(t,win,3,deriv=2,delta=dt)[idx] for t in T])
def eps_cov(dt,sg):
    h=[savgol_coeffs(win,3,deriv=d,delta=dt) for d in (0,1,2)]
    C=np.array([[h[a]@h[b] for b in range(3)] for a in range(3)])*sg**2
    return C   # joint covariance of (theta~, omega~, alpha~) smoothing errors for white angle noise sg
def noisy_feats(S,dt,sg,ii,rng):
    Ts,Ws,As=S; C=eps_cov(dt,sg); L3=np.linalg.cholesky(C+1e-30*np.eye(3)); m=len(ii)
    T=[];W=[];A=[]
    for a in range(3):
        e=rng.standard_normal((m,3))@L3.T
        T.append(Ts[a][ii]+e[:,0]); W.append(Ws[a][ii]+e[:,1]); A.append(As[a][ii]+e[:,2])
    return feats(T,W,A)[0]
R={}
for i in (1,2,3):
    T,_,dt=load(i); L=len(T[0]); idx=np.arange(win,L-win,200); n=len(idx)
    Tsm=[savgol_filter(t,win,3) for t in T]; RES=[t-sm for t,sm in zip(T,Tsm)]; Fb=feats(*smooth3(Tsm,dt,idx))[0]
    Nnat=0
    for _ in range(15):
        nb=L//BL+1; st=rg.integers(0,L-BL,nb); ix=np.concatenate([np.arange(s_,s_+BL) for s_ in st])[:L]
        Tp=[s+r[ix]/np.sqrt(1-h0c) for s,r in zip(Tsm,RES)]; E=feats(*smooth3(Tp,dt,idx))[0]-Fb; Nnat=Nnat+E.T@E/n
    Nnat/=15
    S=smooth3(T,dt,idx); Fc=feats(*S)[0]; Sc=Fc.T@Fc/n; scc=np.sqrt(np.diag(Sc))
    Nc=Nnat/np.outer(scc,scc)+1e-12*np.eye(P); muc,Uc=sl.eigh(Sc/np.outer(scc,scc),Nc)
    R[i]=dict(S=S,dt=dt,n=n,scc=scc,Nnat=Nnat,Nc=Nc,ref=Uc[:,0],Fcs=Fc/scc)
    print("prepared run",i,flush=True)

from scipy.signal import lfilter
SG=0.003; PHI=0.9; K=16
def ar1(L,rng,sg,phi):
    e=rng.standard_normal(L)*sg*np.sqrt(1-phi**2); return lfilter([1],[1,-phi],e)
def filt_rows(e3,dt,idx):  # errors of smoothed angle/velocity/acceleration for noise series e3 (list of 3 arrays) at idx
    return [np.stack([savgol_filter(e,win,3)[idx],savgol_filter(e,win,3,deriv=1,delta=dt)[idx],savgol_filter(e,win,3,deriv=2,delta=dt)[idx]],1) for e in e3]
def feats_from_err(S,errs,ii):    # errs: list over angles of (m,3) arrays aligned with ii
    Ts,Ws,As=S
    return feats([Ts[a][ii]+errs[a][:,0] for a in range(3)],[Ws[a][ii]+errs[a][:,1] for a in range(3)],[As[a][ii]+errs[a][:,2] for a in range(3)])[0]
out=[]
for i in (1,2,3):
    r=R[i]; S=r['S']; dt=r['dt']; n=r['n']; scc=r['scc']; T,_,_=load(i); L=len(T[0]); idx=np.arange(win,L-win,200)
    Tn=[t+ar1(L,rg,0.002,0.99)+ar1(L,rg,0.002,0.5) for t in T]
    Fn=feats(*smooth3(Tn,dt,idx))[0]/scc
    Tsn=[savgol_filter(t,win,3) for t in Tn]; rn=[(t-s_)/np.sqrt(1-h0c) for t,s_ in zip(Tn,Tsn)]
    sig_w=np.mean([np.std(x) for x in rn])
    F0=feats(*S)[0]
    # model W: white Gaussian angle noise with level read from residuals
    Lw=np.linalg.cholesky(eps_cov(dt,sig_w)+1e-30*np.eye(3))
    def errs_white(m,rng): return [rng.standard_normal((m,3))@Lw.T for a in range(3)]
    # model B: block-resampled residuals through the filters (keeps colour)
    pool_err=[]
    for _ in range(K):
        nb=L//BL+1; st=rg.integers(0,L-BL,nb); ix=np.concatenate([np.arange(s_,s_+BL) for s_ in st])[:L]
        pool_err.append(filt_rows([x[ix] for x in rn],dt,idx))     # list over realisations of list over angles of (n,3)
    def errs_block(ii,rng):
        k=rng.integers(0,K,len(ii)); return [np.stack([pool_err[kk][a][i_] for kk,i_ in zip(k,ii)]) for a in range(3)]
    res=dict(run=i,sigma_hat_white=float(sig_w),sigma_true=SG)
    Nnat=r['Nnat']; Ln=np.linalg.cholesky(Nnat/np.outer(scc,scc)+1e-12*np.eye(P))
    ref=r['ref']; Nc=r['Nc']; ca=lambda a:(a@Nc@ref)**2/((a@Nc@a)*(ref@Nc@ref)); pool=np.arange(0,n,4)
    for model in ("block",):
        # noise covariance implied by the model, state dependent through the feature map
        N=0; tot=0
        for _ in range(6):
            ii=np.arange(n); er=errs_white(n,rg) if model=="white" else errs_block(ii,rg)
            E=feats_from_err(S,er,ii)-F0; N=N+E.T@E/n; tot+=1
        Ntot=(Nnat+N/tot)/np.outer(scc,scc)+1e-12*np.eye(P)
        pred=[];meas=[]
        for m in MS:
            ap=[];am=[]
            for _ in range(60):
                ii=rg.choice(pool,m,replace=False)
                er=errs_white(m,rg) if model=="white" else errs_block(ii,rg)
                X=feats_from_err(S,er,ii)/scc+rg.standard_normal((m,P))@Ln.T
                w,U=sl.eigh(X.T@X/m,Ntot); ap.append(ca(U[:,0]))
                Xm=Fn[ii]; w,U=sl.eigh(Xm.T@Xm/m,Ntot); am.append(ca(U[:,0]))
            pred.append(round(float(np.mean(ap)),2)); meas.append(round(float(np.mean(am)),2))
        res[model]=dict(pred=pred,meas=meas,rms=float(np.sqrt(np.mean((np.array(pred)-np.array(meas))**2))),mu0=float(sl.eigh((Fn.T@Fn/n),Ntot,eigvals_only=True)[0]))
        print(i,model,res[model],flush=True)
    out.append(res); json.dump(out,open("tri_color_mix.json","w"))
