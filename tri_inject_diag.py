import numpy as np, scipy.linalg as sl
from scipy.signal import savgol_filter, savgol_coeffs
from tri import *
rg=np.random.default_rng(91); win=301; sg=0.003
h0=savgol_coeffs(win,3,deriv=0)[win//2]
def smooth3(T,dt,idx):
    return ([savgol_filter(t,win,3)[idx] for t in T],[savgol_filter(t,win,3,deriv=1,delta=dt)[idx] for t in T],[savgol_filter(t,win,3,deriv=2,delta=dt)[idx] for t in T])
T,_,dt=load(1); L=len(T[0]); idx=np.arange(win,L-win,200); n=len(idx)
Tsm=[savgol_filter(t,win,3) for t in T]; Fb=feats(*smooth3(Tsm,dt,idx))[0]
Tn=[t+sg*rg.standard_normal(L) for t in T]; Fn=feats(*smooth3(Tn,dt,idx))[0]
Ninj=0
for _ in range(10):
    Tp=[s+sg*rg.standard_normal(L) for s in Tsm]; E=feats(*smooth3(Tp,dt,idx))[0]-Fb; Ninj+=E.T@E/n
Ninj/=10
S=Fn.T@Fn/n; sc=np.sqrt(np.diag(S)); Ss=S/np.outer(sc,sc); Ns=Ninj/np.outer(sc,sc)+1e-12*np.eye(P)
mu,U=sl.eigh(Ss,Ns); lam=mu[1:]-1; lam=lam[lam>0]; A1=np.sum(1+1/lam);A2=np.sum((1+1/lam)**2); ref=U[:,0]
ca=lambda a:(a@Ns@ref)**2/((a@Ns@a)*(ref@Ns@ref)); Fs=Fn/sc
print("A1 %.0f A2 %.0f mu0 %.2f  smallest lam %s"%(A1,A2,mu[0],np.round(np.sort(lam)[:5],2)))
def curve(sampler,ms=(30,60,100,200,400)):
    out=[]
    for m in ms:
        al=[]
        for _ in range(80):
            X=sampler(m); Sn=X.T@X/m; w,Un=sl.eigh(Sn,Ns); al.append(ca(Un[:,0]))
        out.append(round(float(np.mean(al)),2))
    return out
for step in (1,4,16):
    pool=np.arange(0,n,step); print("real rows, spacing x%d (pool %d):"%(step,len(pool)),curve(lambda m:Fs[rg.choice(pool,min(m,len(pool)),replace=False)] if m<=len(pool) else Fs[rg.choice(pool,len(pool),replace=False)],ms=(30,60,100,200,400) if step<16 else (30,60,100,180)))
# Gaussian surrogate: rows ~ N(0, Ss) (same population covariance)
Lc=np.linalg.cholesky(Ss+1e-12*np.eye(P))
print("Gaussian surrogate N(0,S):",curve(lambda m:rg.standard_normal((m,P))@Lc.T))
# kurtosis of whitened rows in generalized eigenbasis
Z=Fs@U; print("excess kurtosis of generalized-eigen coordinates: median %.1f max %.1f"%(np.median(((Z-Z.mean(0))**4).mean(0)/Z.var(0)**2-3),np.max(((Z-Z.mean(0))**4).mean(0)/Z.var(0)**2-3)))

Zc=Z-Z.mean(0)
def marg(m):
    Y=np.stack([Zc[rg.integers(0,n,m),j] for j in range(P)],1)   # independent marginal resampling keeps each coordinate's tails
    return Y@np.linalg.inv(U)
print("marginal-bootstrap surrogate (keeps tails, removes dependence):",curve(marg))
