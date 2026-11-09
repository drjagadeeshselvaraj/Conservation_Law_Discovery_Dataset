import numpy as np, scipy.linalg as sl, json, pickle
from scipy.signal import savgol_filter, savgol_coeffs
from tri import *
rg=np.random.default_rng(121); win=201; BL=1000; MS=(30,60,100,200,400,700)
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
def Ntot_of(j,sg,rng,draws=6):
    r=R[j]; n=r['n']; ii=np.arange(n); F0=feats(*r['S'])[0]; N=0
    for _ in range(draws):
        E=noisy_feats(r['S'],r['dt'],sg,ii,rng)-F0; N=N+E.T@E/n
    N/=draws; return (r['Nnat']+N)/np.outer(r['scc'],r['scc'])+1e-12*np.eye(P)
def curve(j,sg,Ntot,rng,reps=60):
    r=R[j]; pool=np.arange(0,r['n'],4); Nc=r['Nc']; ref=r['ref']; res=[]
    ca=lambda a:(a@Nc@ref)**2/((a@Nc@a)*(ref@Nc@ref))
    for m in MS:
        al=[]
        for _ in range(reps):
            ii=rng.choice(pool,m,replace=False)
            X=noisy_feats(r['S'],r['dt'],sg,ii,rng)/r['scc']+rng.standard_normal((m,P))@np.linalg.cholesky(r['Nnat']/np.outer(r['scc'],r['scc'])+1e-12*np.eye(P)).T
            w,U=sl.eigh(X.T@X/m,Ntot); al.append(ca(U[:,0]))
        res.append(round(float(np.mean(al)),2))
    return res
out={}
for sg in (0.0003,0.001,0.003):
    Nt={j:Ntot_of(j,sg,rg) for j in (1,2,3)}
    for j in (1,2,3):
        out[f"{j}_{sg}"]=curve(j,sg,Nt[j],rg); print("state-dependent plug-in from run",j,"sigma",sg,out[f"{j}_{sg}"],flush=True)
    json.dump(out,open("tri_state_w201.json","w"))
