import numpy as np, sys, json, tb
from scipy.linalg import eigh
f=float(sys.argv[1]); K=4
X=np.load("X.npy"); rg=np.random.default_rng(7)
F0=tb.feats(X); sc=np.sqrt((F0**2).mean(0))
U,s,Vt=np.linalg.svd(F0/sc); Cs=Vt[-K:].T            # exact-law subspace in scaled coordinates (p x K)
sd=f*X.std(0)
Xn=X+rg.standard_normal(X.shape)*sd
F=tb.feats(Xn)
D=[]
for _ in range(20):
    D.append(tb.feats(Xn+rg.standard_normal(X.shape)*sd)-F)
D=np.vstack(D); N=D.T@D/len(D); Ns=N/np.outer(sc,sc)+1e-14*np.eye(25)
Fs=F/sc; S=Fs.T@Fs/len(Fs); mu,Vg=eigh(S,Ns)
lam=mu[K:]-1; neg=int((lam<=0).sum()); lam=lam[lam>0]
A1=float(np.sum(1+1/lam)); A2=float(np.sum((1+1/lam)**2))
print(f"f={f}: mu[:{K+2}]={np.round(mu[:K+2],3)} nneg={neg} A1={A1:.0f} A2={A2:.0f}")
# whitened coordinates
w,Q=np.linalg.eigh(Ns); Nh=Q@np.diag(w**.5)@Q.T; Nih=Q@np.diag(w**-.5)@Q.T
W,_=np.linalg.qr(Nh@Cs)                                # law subspace after whitening
def ov(V): return float(np.sum((W.T@V)**2)/K)
rows=[]
for m in (100,200,400,800,1600,3200,6400,8000):
    aw=[];ap=[]
    for _ in range(30):
        ii=rg.choice(len(X),m,replace=False); Sn=Fs[ii].T@Fs[ii]/m
        Sw=Nih@Sn@Nih; aw.append(ov(np.linalg.eigh(Sw)[1][:,:K]))
        # plain: bottom-K of un-whitened S, compared in the same N-metric
        Vp=np.linalg.eigh(Sn)[1][:,:K]; Vp,_=np.linalg.qr(Nh@Vp); ap.append(ov(Vp))
    pred=(1-A2/m)/(1-A1/m) if m>A2 else 0.0
    rows.append((m,np.mean(aw),pred,np.mean(ap))); print(f"m={m:5d} whitened {np.mean(aw):.3f} formula {pred:.3f} plain {np.mean(ap):.3f}")
json.dump(dict(f=f,A1=A1,A2=A2,mu=list(map(float,mu[:K+2])),rows=rows),open(f"res_{f}.json","w"))
