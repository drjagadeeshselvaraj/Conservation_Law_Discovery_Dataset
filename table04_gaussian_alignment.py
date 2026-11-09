import numpy as np, scipy.linalg as sl
from lawsample import *
rg=np.random.default_rng(5); p=200; k=1
tests={"equal lam=1":np.full(p-k,1.0),"two levels 0.8/3":np.r_[np.full(100,0.8),np.full(99,3.0)],"graded":np.linspace(0.8,4,p-k)}
for name,lam in tests.items():
    A1,A2=constants(lam); Q,_=np.linalg.qr(rg.standard_normal((p,p)))
    ev=np.r_[1.0,1+lam]; Sig=Q@np.diag(ev)@Q.T; c=Q[:,0]; Lc=np.linalg.cholesky(Sig)
    row=[]
    for f in (1.2,1.5,2.0,3.0):
        n=int(f*A2); al=[]
        for _ in range(20):
            X=rg.standard_normal((n,p))@Lc.T; w,V=np.linalg.eigh(X.T@X/n); al.append((V[:,0]@c)**2)
        row.append((n,round(float(np.mean(al)),2),round(alignment(n,lam),2)))
    print(name,"A2=%.0f"%A2,"(n, measured, predicted):",row)
print("samples for 0.9, equal lam=1, p=200:",round(samples_for_alignment(0.9,np.full(199,1.0))))
