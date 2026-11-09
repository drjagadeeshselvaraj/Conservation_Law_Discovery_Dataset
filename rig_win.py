import numpy as np, scipy.linalg as sl, json, sys
from scipy.signal import savgol_filter, savgol_coeffs
import tri
from tri import feats, P
i=int(sys.argv[1]); BL=1000; rg=np.random.default_rng(40+i)
T,_,dt=tri.load(i); L=len(T[0]); res={}
for win in (61,101,201,301,601):
    h0=savgol_coeffs(win,3,deriv=0)[win//2]
    def sm3(X,idx): return ([savgol_filter(x,win,3)[idx] for x in X],[savgol_filter(x,win,3,deriv=1,delta=dt)[idx] for x in X],[savgol_filter(x,win,3,deriv=2,delta=dt)[idx] for x in X])
    idx=np.arange(win,L-win,200); n=len(idx)
    F=feats(*sm3(T,idx))[0]
    Tsm=[savgol_filter(x,win,3) for x in T]; RES=[x-s for x,s in zip(T,Tsm)]; Fb=feats(*sm3(Tsm,idx))[0]; Nb=0
    for _ in range(10):
        nb=L//BL+1; st=rg.integers(0,L-BL,nb); ix=np.concatenate([np.arange(s_,s_+BL) for s_ in st])[:L]
        Tp=[s+r[ix]/np.sqrt(1-h0) for s,r in zip(Tsm,RES)]; E=feats(*sm3(Tp,idx))[0]-Fb; Nb=Nb+E.T@E/n
    Nb/=10
    sc=np.sqrt(np.diag(F.T@F/n)); S=F.T@F/n/np.outer(sc,sc)
    mb=sl.eigh(S,Nb/np.outer(sc,sc)+1e-12*np.eye(P),eigvals_only=True)
    res[win]=float(mb[0]); print("rig run",i,"win",win,"mu0_block %.2f"%mb[0],flush=True)
json.dump(res,open(f"rig_win_{i}.json","w"))
