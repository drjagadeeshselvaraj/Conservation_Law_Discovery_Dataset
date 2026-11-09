import numpy as np, sys, json
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
def acc(s):
    q=s[:,1:3]; a=np.zeros((3,2))
    for i in range(3):
        for j in range(3):
            if i!=j:
                d=q[j]-q[i]; a[i]+=d/np.linalg.norm(d)**3
    return a
def upd(t,y):
    s=y.reshape(3,4); out=np.zeros_like(s); out[:,0:2]=s[:,2:4]; out[:,2:4]=acc(np.c_[np.ones(3),s[:,:2]]); return out.ravel()
def rot(p,th): c,s=np.cos(th),np.sin(th); return np.array([[c,-s],[s,c]])@p
def config(rg,nu=0.2):
    p1=2*rg.random(2)-1; r=rg.random()*0.3+0.9; p1*=r/np.linalg.norm(p1)
    P=[p1,rot(p1,2*np.pi/3),rot(rot(p1,2*np.pi/3),2*np.pi/3)]
    v1=rot(p1,np.pi/2)/r**1.5*np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))
    V=[v1,rot(v1,2*np.pi/3),rot(rot(v1,2*np.pi/3),2*np.pi/3)]
    V=[v*(1+nu*(2*rg.random(2)-1)) for v in V]
    q=np.array(P); v=np.array(V); v-=v.mean(0)          # total momentum 0
    L=np.sum(q[:,0]*v[:,1]-q[:,1]*v[:,0]); LT=float(__import__("os").environ.get("LT","2.5")); v*=LT/L
    v=v+np.array([0.3,-0.2])   # common drift: total momentum (0.9,-0.6), same for every orbit
    return np.c_[q,v]
def _one(seed,npt=20,T=5.0):
    from scipy.integrate import solve_ivp
    rg=np.random.default_rng(seed)
    s=config(rg)
    def close(t,y):
        q=y.reshape(3,4)[:,:2]
        return min(np.linalg.norm(q[0]-q[1]),np.linalg.norm(q[0]-q[2]),np.linalg.norm(q[1]-q[2]))-0.15
    close.terminal=True
    te=np.sort(rg.random(npt)*T)
    sol=solve_ivp(upd,[0,T],s.ravel(),t_eval=te,rtol=1e-8,atol=1e-10,events=close)
    if sol.status!=0 or sol.y.shape[1]<npt: return None
    return sol.y.T.reshape(npt,12)
def orbits(norb=400,npt=20,T=5.0,seed0=1000):
    from multiprocessing import Pool
    with Pool() as pl: res=pl.map(_one,range(seed0,seed0+int(norb*1.5)))
    res=[r for r in res if r is not None][:norb]
    print("orbits kept",len(res),flush=True)
    return np.vstack(res)
def feats(X):
    x=X[:,[0,4,8]]; y=X[:,[1,5,9]]; vx=X[:,[2,6,10]]; vy=X[:,[3,7,11]]
    cols=[np.ones(len(X))]+[vx[:,i] for i in range(3)]+[vy[:,i] for i in range(3)]
    cols+=[x[:,i]*vy[:,j] for i in range(3) for j in range(3)]
    cols+=[y[:,i]*vx[:,j] for i in range(3) for j in range(3)]
    return np.stack(cols,1)
if __name__=="__main__":
    rg=np.random.default_rng(1)
    X=orbits(rg); print(X.shape)
    F0=feats(X); sc=np.sqrt((F0**2).mean(0)); s=np.linalg.svd(F0/sc,compute_uv=False); print("sing vals smallest 6:",s[-6:])
    np.save("X.npy",X)
