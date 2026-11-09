import sys
sys.path.insert(0,'/tmp/claude-0/-home-claude/1bdb3634-9951-583d-9a91-beca2986667c/scratchpad/stubs')
sys.path.insert(0,'/tmp/claude-0/-home-claude/1bdb3634-9951-583d-9a91-beca2986667c/scratchpad/pysindy_src')
import numpy as np, scipy.linalg as sl
from scipy.optimize import minimize
import pysindy as ps

# ---------- SINDy-PI-style implicit null vector with PySINDy's STLSQ, threshold tuned on held-out rows
def _stlsq_vec(Fm, j, tau, alpha=1e-5):
    P=Fm.shape[1]; idx=[q for q in range(P) if q!=j]
    o=ps.STLSQ(threshold=tau,alpha=alpha,max_iter=10); o.fit(Fm[:,idx],Fm[:,[j]]*-1.0)
    v=np.zeros(P); v[idx]=o.coef_.ravel(); v[j]=1.0; return v
def sindy_tuned(Fm, taus=(0.01,0.03,0.1,0.3), rg=None):
    rg=rg or np.random.default_rng(0); m=len(Fm); perm=rg.permutation(m); A,B=perm[:m//2],perm[m//2:]
    best=None
    for tau in taus:
        for j in range(Fm.shape[1]):
            v=_stlsq_vec(Fm[A],j,tau)
            r=np.linalg.norm(Fm[B]@v)/np.linalg.norm(v)
            if best is None or r<best[0]: best=(r,j,tau)
    _,j,tau=best; v=_stlsq_vec(Fm,j,tau); return v/np.linalg.norm(v),(j,tau)

# ---------- neural invariant with held-out selection of width
def _nn_fit(X,Xd,H,rg,restarts=2,iters=150):
    m,d=X.shape
    def unpack(th): return th[:H*d].reshape(H,d),th[H*d:H*d+H],th[H*d+H:H*d+2*H]
    def lg(th):
        W1,b1,w2=unpack(th); a=X@W1.T+b1; h=np.tanh(a); s=1-h*h; B=Xd@W1.T; G=s*B
        Q=h@w2; Qd=G@w2; Qm=Q.mean(); den=((Q-Qm)**2).mean()+1e-6; num=(Qd**2).mean(); L=num/den
        gQd=2*Qd/(m*den); gQ=-num/den**2*2*(Q-Qm)/m; gw2=G.T@gQd+h.T@gQ
        dG=np.outer(gQd,w2); ds=dG*B; dB=dG*s; dh=-2*h*ds+np.outer(gQ,w2); da=dh*s
        return L,np.concatenate([(da.T@X+dB.T@Xd).ravel(),da.sum(0),gw2])
    best=None
    for _ in range(restarts):
        th0=np.concatenate([rg.standard_normal(H*d)*0.7,rg.standard_normal(H)*0.3,rg.standard_normal(H)*0.5])
        r=minimize(lg,th0,jac=True,method="L-BFGS-B",options=dict(maxiter=iters))
        if best is None or r.fun<best.fun: best=r
    return unpack(best.x)
def _nn_score(par,X,Xd):
    W1,b1,w2=par; h=np.tanh(X@W1.T+b1); Q=h@w2; Qd=((1-h*h)*(Xd@W1.T))@w2
    return (Qd**2).mean()/(Q.var()+1e-6)
def nn_tuned(X,Xd,rg,widths=(8,24,48)):
    m=len(X); perm=rg.permutation(m); A,B=perm[:m//2],perm[m//2:]; best=None
    for H in widths:
        sc=_nn_score(_nn_fit(X[A],Xd[A],H,rg),X[B],Xd[B])
        if best is None or sc<best[0]: best=(sc,H)
    W1,b1,w2=_nn_fit(X,Xd,best[1],rg)
    return (lambda Z: np.tanh(Z@W1.T+b1)@w2),best[1]

# ---------- kernel invariant with held-out selection of bandwidth and regularisation
def _k_fit(X,Xd,bw,reg):
    m=len(X); d2=((X[:,None,:]-X[None,:,:])**2).sum(-1); ell2=bw*np.median(d2[d2>0]); K=np.exp(-d2/(2*ell2))
    Dm=-K*((X*Xd).sum(1)[:,None]-Xd@X.T)/ell2
    Kc=K-K.mean(0)-K.mean(1)[:,None]+K.mean(); A=Dm.T@Dm/m
    B=Kc.T@Kc/m+reg*np.trace(Kc.T@Kc/m)/m*np.eye(m)
    w,V=sl.eigh(A+1e-12*np.trace(A)/m*np.eye(m),B); return X,ell2,V[:,0]
def _k_eval(par,Z,Zd):
    X,ell2,al=par; d=((Z[:,None,:]-X[None,:,:])**2).sum(-1); K=np.exp(-d/(2*ell2))
    Q=K@al; Qd=(-K*((Z*Zd).sum(1)[:,None]-Zd@X.T)/ell2)@al; return Q,Qd
def kernel_tuned(X,Xd,rg,bws=(0.5,1.0,2.0),regs=(1e-4,1e-3,1e-2)):
    m=len(X); perm=rg.permutation(m); A,B=perm[:m//2],perm[m//2:]; best=None
    for bw in bws:
        for reg in regs:
            Q,Qd=_k_eval(_k_fit(X[A],Xd[A],bw,reg),X[B],Xd[B]); sc=(Qd**2).mean()/(Q.var()+1e-12)
            if best is None or sc<best[0]: best=(sc,bw,reg)
    par=_k_fit(X,Xd,best[1],best[2]); return (lambda Z:_k_eval(par,Z,np.zeros_like(Z))[0]),(best[1],best[2])
