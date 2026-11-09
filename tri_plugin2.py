import numpy as np, scipy.linalg as sl, json
from scipy.signal import savgol_filter, savgol_coeffs
from tri import *
rg=np.random.default_rng(111); win=301; BL=1000; out={}
h0=savgol_coeffs(win,3,deriv=0)[win//2]
def smooth3(T,dt,idx):
    return ([savgol_filter(t,win,3)[idx] for t in T],[savgol_filter(t,win,3,deriv=1,delta=dt)[idx] for t in T],[savgol_filter(t,win,3,deriv=2,delta=dt)[idx] for t in T])
MS=(30,60,100,200,400)
for i in (1,2,3):
    T,_,dt=load(i); L=len(T[0]); idx=np.arange(win,L-win,200); n=len(idx)
    Tsm=[savgol_filter(t,win,3) for t in T]; RES=[t-sm for t,sm in zip(T,Tsm)]; Fb=feats(*smooth3(Tsm,dt,idx))[0]
    Nnat=0
    for _ in range(15):
        nb=L//BL+1; st=rg.integers(0,L-BL,nb); ix=np.concatenate([np.arange(s_,s_+BL) for s_ in st])[:L]
        Tp=[s+r[ix]/np.sqrt(1-h0) for s,r in zip(Tsm,RES)]; E=feats(*smooth3(Tp,dt,idx))[0]-Fb; Nnat=Nnat+E.T@E/n
    Nnat/=15
    Fc=feats(*smooth3(T,dt,idx))[0]; Sc=Fc.T@Fc/n; scc=np.sqrt(np.diag(Sc)); Fcs=Fc/scc
    Nc=Nnat/np.outer(scc,scc)+1e-12*np.eye(P); muc,Uc=sl.eigh(Sc/np.outer(scc,scc),Nc); refc=Uc[:,0]
    ca=lambda a:(a@Nc@refc)**2/((a@Nc@a)*(refc@Nc@refc)); pool=np.arange(0,n,4)
    for sg in (0.001,0.003):
        Tn=[t+sg*rg.standard_normal(L) for t in T]; Fn=feats(*smooth3(Tn,dt,idx))[0]/scc
        Ereal=Fn-Fcs                                   # noise rows actually present in the noisy data (tied to their signal rows)
        Ninj=0
        for _ in range(10):
            Tp=[s+sg*rg.standard_normal(L) for s in Tsm]; E=feats(*smooth3(Tp,dt,idx))[0]-Fb; Ninj=Ninj+E.T@E/n
        Ntot=(Nnat+Ninj/10)/np.outer(scc,scc)+1e-12*np.eye(P)
        # kurtosis of noise rows in whitened coordinates
        W=np.linalg.inv(np.linalg.cholesky(Ntot)); Zw=Ereal@W.T; ek=((Zw-Zw.mean(0))**4).mean(0)/Zw.var(0)**2-3
        def curve(rowfn):
            r=[]
            for m in MS:
                al=[]
                for _ in range(60):
                    X=rowfn(m); w,U=sl.eigh(X.T@X/m,Ntot); al.append(ca(U[:,0]))
                r.append(round(float(np.mean(al)),2))
            return r
        def tied(m): ii=rg.choice(pool,m,replace=False); return Fcs[ii]+Ereal[ii]
        def shuf(m): ii=rg.choice(pool,m,replace=False); jj=rg.choice(pool,m,replace=False); return Fcs[ii]+Ereal[jj]
        Lt=np.linalg.cholesky(Ntot)
        def gauss(m): ii=rg.choice(pool,m,replace=False); return Fcs[ii]+rg.standard_normal((m,P))@Lt.T
        res=dict(real=curve(tied),shuffled_noise=curve(shuf),gaussian_noise=curve(gauss),noise_excess_kurtosis_median=float(np.median(ek)),noise_excess_kurtosis_max=float(ek.max()))
        print(i,sg,res,flush=True); out[f"{i}_{sg}"]=res; json.dump(out,open("tri_plugin2.json","w"))
