import numpy as np
from scipy.optimize import brentq
# population: atom a=1/(1+s) weight kappa, atom 1 weight 1-kappa (rescaled so bulk=1)
def separates(g,s,kappa):
    a=1/(1+s)
    m=np.linspace(-1/a+1e-6,-1-1e-6,20001)
    zp=1/m**2-g*((1-kappa)*1/(1+m)**2+kappa*a**2/(1+a*m)**2)
    return zp.max()>0
def nstar(s,kappa):
    # smallest n/p (largest g<1) with separation: bisection on g
    lo,hi=1e-6,0.999
    if not separates(lo,s,kappa): return np.inf
    for _ in range(60):
        mid=(lo+hi)/2
        if separates(mid,s,kappa): lo=mid
        else: hi=mid
    return 1/lo
for s in [0.5,1,2,4]:
    print("s=",s,[round(nstar(s,k),2) for k in [0.0005,0.02,0.05,0.1,0.2,0.3]], "kappa->0:",round((1+1/s)**2,2))
# simulation check for s=1
rng=np.random.default_rng(2)
def gapcheck(p,r,s,kappa):
    k=int(kappa*p);n=int(r*p)
    ev=np.ones(p);ev[:k]=1;ev[k:]=1+s
    X=rng.standard_normal((n,p))*np.sqrt(ev)
    w=np.linalg.eigvalsh(X.T@X/n)
    return w[k]-w[k-1], w[k-1], w[k]
p=1000
for kappa in [0.1,0.3]:
    ns=nstar(1,kappa); print("kappa",kappa,"n*/p",round(ns,2))
    for r in [ns*0.7,ns*0.9,ns*1.1,ns*1.4]:
        print(" r=%.2f gap=%.3f (lower edge-top, upper-bottom: %.3f %.3f)"%((r,)+gapcheck(p,r,1,kappa)))
