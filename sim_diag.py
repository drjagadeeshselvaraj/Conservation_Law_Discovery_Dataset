import numpy as np, scipy.linalg as sl, json, sys
from scipy.signal import savgol_filter, savgol_coeffs
import tri, trisim
import sim_gen_lib as G
win=301; rg=np.random.default_rng(11)
def sm3(T,dt,idx): return ([savgol_filter(t,win,3)[idx] for t in T],[savgol_filter(t,win,3,deriv=1,delta=dt)[idx] for t in T],[savgol_filter(t,win,3,deriv=2,delta=dt)[idx] for t in T])
def stats(name,T,dt,sg):
    L=len(T[0]); idx=np.arange(win,L-win,200); n=len(idx)
    S=sm3(T,dt,idx); F,_=tri.feats(*S)[0],0; sc=F.std(0)+1e-12; Fs=(F-0)/np.sqrt((F**2).mean(0))
    Sm=Fs.T@Fs/n; ev=np.sort(np.linalg.eigvalsh(Sm))[::-1]; pr=ev.sum()**2/(ev**2).sum()
    kurt=np.median([np.mean((Fs[:,k]-Fs[:,k].mean())**4)/np.var(Fs[:,k])**2 for k in range(Fs.shape[1])])
    # row-wise noise energy variation
    Tsm=[savgol_filter(t,win,3) for t in T]; Fb=tri.feats(*sm3(Tsm,dt,idx))[0]
    Es=[]
    for _ in range(12):
        Tp=[s+sg*rg.standard_normal(L) for s in Tsm]; E=(tri.feats(*sm3(Tp,dt,idx))[0]-Fb)/np.sqrt((F**2).mean(0)); Es.append((E**2).sum(1))
    Es=np.mean(Es,0); cv=Es.std()/Es.mean()
    ang=[np.std(t) for t in Tsm]; vel=[np.std(savgol_filter(t,win,3,deriv=1,delta=dt)) for t in T]
    print(name,"PR(eff dim) %.1f"%pr,"median kurt %.2f"%kurt,"row-noise CV %.2f"%cv,"ang sd",np.round(ang,2),"vel sd",np.round(vel,1),flush=True)
for i in (1,2,3):
    T,_,dt=tri.load(i); stats(f"rig run{i}",T,dt,0.003)
for i in (1,2,3):
    z=np.load(f"sim_run{i}.npz"); T=[z["T1"],z["T2"],z["T3"]]; stats(f"sim run{i}",T,1e-4,0.003)
