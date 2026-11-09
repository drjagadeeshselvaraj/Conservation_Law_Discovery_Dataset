# Table 21 (second experimental double pendulum, mu_0 by stretch), REBUILT from the method in the text.
# Library: 15-term double-pendulum library (dp_lib.feat), rows = time derivatives of the features,
# Savitzky-Golay window 31 (3rd order), rows every 0.01 s, noise cov by parametric resampling with the reported sd.
# Run from the folder that contains vid/ (dpm/).
import numpy as np, sys
from scipy.linalg import eigh
sys.path.insert(0,'.')
from dp_lib import load, feat, smooth
W=31; STEP=5; rg=np.random.default_rng(0)
def rows(t,th,dt):
    s=smooth(th,dt,W); T=s[0]; Om=s[1]; A=s[2]
    D,F=feat(T,Om,A); return D[::STEP],t[::STEP]
for tr in (1,2,3):
    t,th,sd=load(tr); dt=t[1]-t[0]
    R0,tt=rows(t,th,dt)
    Ds=[]
    for _ in range(3):
        Rn,_=rows(t,th+sd*rg.standard_normal(th.shape),dt); Ds.append(Rn-R0)
    out=[]
    for a,b in ((5,15),(15,25),(25,35),(35,45),(45,55)):
        m=(tt>=a)&(tt<b)
        # noise cov within the stretch: the same rows, per stretch
        N=np.vstack([d[m] for d in Ds]); 
        Rw=R0[m]; cs=np.sqrt((Rw**2).mean(0))+1e-12
        S=(Rw/cs).T@(Rw/cs)/len(Rw); Nn=(N/cs).T@(N/cs)/len(N)
        w=eigh(S,Nn+1e-12*np.trace(Nn)/len(Nn)*np.eye(len(Nn)),eigvals_only=True)
        out.append('%d-%d: %.2f (%.1f)'%(a,b,w[0],w[1]))
    print('Trial',tr,' | '.join(out))
