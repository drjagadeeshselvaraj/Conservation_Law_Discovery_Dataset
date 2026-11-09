import numpy as np, scipy.linalg as sl, json
from scipy.signal import savgol_filter, savgol_coeffs
import sim_gen_lib as G
from trisim import feats, P
win=301; BL=1000; rg=np.random.default_rng(3); h0=savgol_coeffs(win,3,deriv=0)[win//2]; sg=3e-5
def sm3(T,dt,idx): return ([savgol_filter(t,win,3)[idx] for t in T],[savgol_filter(t,win,3,deriv=1,delta=dt)[idx] for t in T],[savgol_filter(t,win,3,deriv=2,delta=dt)[idx] for t in T])
out={}
for c in (0.0,0.002):
    Y,t,dt=G.run((1.2,0.9,0.5),c,rg); L=len(t); idx=np.arange(win,L-win,200); n=len(idx)
    Yc=[Y[k] for k in range(3)]
    # exact features from the true state and derivatives
    Wt=[Y[3+k] for k in range(3)]; 
    At=[np.gradient(Y[3+k],dt) for k in range(3)]
    Fex=feats([y[idx] for y in Yc],[w[idx] for w in Wt],[a[idx] for a in At])[0]
    Fsc=feats(*sm3(Yc,dt,idx))[0]                       # smoothed, noiseless
    T=[y+sg*rg.standard_normal(L) for y in Yc]; F=feats(*sm3(T,dt,idx))[0]
    # true noise covariance by Monte Carlo over noise draws
    Nt=0
    for _ in range(25):
        Tn=[y+sg*rg.standard_normal(L) for y in Yc]; E=feats(*sm3(Tn,dt,idx))[0]-Fsc; Nt=Nt+E.T@E/n
    Nt/=25
    # block estimate from the single noisy run
    Tsm=[savgol_filter(x,win,3) for x in T]; RES=[x-s for x,s in zip(T,Tsm)]; Fb=feats(*sm3(Tsm,dt,idx))[0]; Nb=0
    for _ in range(15):
        nb=L//BL+1; st=rg.integers(0,L-BL,nb); ix=np.concatenate([np.arange(s_,s_+BL) for s_ in st])[:L]
        Tp=[s+r[ix]/np.sqrt(1-h0) for s,r in zip(Tsm,RES)]; E=feats(*sm3(Tp,dt,idx))[0]-Fb; Nb=Nb+E.T@E/n
    Nb/=15
    sc=np.sqrt(np.diag(F.T@F/n)); S=F.T@F/n/np.outer(sc,sc); r=1e-12*np.eye(P)
    mu_b,Ub=sl.eigh(S,Nb/np.outer(sc,sc)+r); mu_t,Ut=sl.eigh(S,Nt/np.outer(sc,sc)+r)
    # smoothing bias along the null direction: noiseless smoothed vs exact features
    c0=Ut[:,0]/sc; bias=(Fsc-Fex)@c0; nvar=c0@Nt@c0
    Sx=(Fex.T@Fex/n)/np.outer(sc,sc); mu_x=sl.eigh(Sx,Nt/np.outer(sc,sc)+r,eigvals_only=True)[0]
    out[c]=dict(mu0_block=float(mu_b[0]),mu0_trueN=float(mu_t[0]),mu0_exact_features=float(mu_x),bias_var_over_noise=float(np.mean(bias**2)/nvar),trace_ratio=float(np.trace(Nb)/np.trace(Nt)))
    print("friction",c,out[c],flush=True)
json.dump(out,open("sim_mu0true.json","w"))
