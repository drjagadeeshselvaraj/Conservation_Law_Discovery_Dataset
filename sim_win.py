import numpy as np, scipy.linalg as sl, json, sys
from scipy.signal import savgol_filter, savgol_coeffs
import sim_gen_lib as G
from trisim import feats, P
IC=eval(sys.argv[1]); tag=sys.argv[2]; sg=3e-5; BL=1000
rg=np.random.default_rng(int(sys.argv[3]))
Y,t,dt=G.run(IC,0.002,rg,Tend=30.0); L0=len(t); Yc=[Y[k] for k in range(3)]
Wt=[Y[3+k] for k in range(3)]; At=[np.gradient(Y[3+k],dt) for k in range(3)]
res={}
for win in (61,101,201,301,601):
    h0=savgol_coeffs(win,3,deriv=0)[win//2]
    def sm3(T,idx): return ([savgol_filter(x,win,3)[idx] for x in T],[savgol_filter(x,win,3,deriv=1,delta=dt)[idx] for x in T],[savgol_filter(x,win,3,deriv=2,delta=dt)[idx] for x in T])
    L=L0; idx=np.arange(win,L-win,100); n=len(idx)
    Fex=feats([y[idx] for y in Yc],[w[idx] for w in Wt],[a[idx] for a in At])[0]
    Fsc=feats(*sm3(Yc,idx))[0]
    T=[y+sg*rg.standard_normal(L) for y in Yc]; F=feats(*sm3(T,idx))[0]
    Nt=0
    for _ in range(10):
        Tn=[y+sg*rg.standard_normal(L) for y in Yc]; E=feats(*sm3(Tn,idx))[0]-Fsc; Nt=Nt+E.T@E/n
    Nt/=10
    Tsm=[savgol_filter(x,win,3) for x in T]; RES=[x-s for x,s in zip(T,Tsm)]; Fb=feats(*sm3(Tsm,idx))[0]; Nb=0
    for _ in range(8):
        nb=L//BL+1; st=rg.integers(0,L-BL,nb); ix=np.concatenate([np.arange(s_,s_+BL) for s_ in st])[:L]
        Tp=[s+r[ix]/np.sqrt(1-h0) for s,r in zip(Tsm,RES)]; E=feats(*sm3(Tp,idx))[0]-Fb; Nb=Nb+E.T@E/n
    Nb/=8
    sc=np.sqrt(np.diag(F.T@F/n)); S=F.T@F/n/np.outer(sc,sc); r=1e-12*np.eye(P)
    mb,Ub=sl.eigh(S,Nb/np.outer(sc,sc)+r); mt,Ut=sl.eigh(S,Nt/np.outer(sc,sc)+r)
    c0=Ut[:,0]/sc; bias=(Fsc-Fex)@c0; nv=c0@Nt@c0
    res[win]=dict(mu0_block=float(mb[0]),mu0_true=float(mt[0]),bias_over_noise=float(np.mean(bias**2)/nv),trace_ratio=float(np.trace(Nb)/np.trace(Nt)))
    print(tag,win,res[win],flush=True)
    json.dump(res,open(f"sim_win_{tag}.json","w"))
