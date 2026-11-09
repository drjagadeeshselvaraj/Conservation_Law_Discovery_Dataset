import numpy as np
rng=np.random.default_rng(7)
def run(p,k,lams,n,reps=20):
    t=np.concatenate([np.ones(k),1+lams]); out=[];ev=[]
    for _ in range(reps):
        X=rng.standard_normal((n,p))*np.sqrt(t)
        S=X.T@X/n; w,V=np.linalg.eigh(S)
        ev.append(w[:k].mean()); out.append((V[:k,:k] if False else (V[:k,:k]**2).sum())/k) # C = first k coords; bottom-k vecs are first k columns
    return np.mean(ev),np.mean(out)
for p,k,lam,ratio in [(400,1,1.0,1.6),(400,2,1.0,1.3),(400,3,2.0,2.0),(400,1,'mix',1.5)]:
    if lam=='mix': lams=np.where(np.arange(p-k)%2==0,0.5,3.0)
    else: lams=np.full(p-k,lam)
    A1=(1+1/lams).sum(); A2=((1+1/lams)**2).sum(); n=int(ratio*A2)
    # alignment: use eigenvectors of smallest k eigenvalues, overlap with first k coords
    t=np.concatenate([np.ones(k),1+lams]); evs=[];rho=[]
    for _ in range(20):
        X=rng.standard_normal((n,p))*np.sqrt(t); S=X.T@X/n; w,V=np.linalg.eigh(S)
        evs.append(w[:k].mean()); rho.append((V[:k,:][:, :k]**0).sum()*0+ (V[:k,:k]**2).sum()/k if False else (V[:k,:]**2).sum()/k)
    # overlap of bottom-k eigenvectors (columns 0..k-1 of V) with coords 0..k-1: rows 0..k-1
    rho=[]
    for _ in range(20):
        X=rng.standard_normal((n,p))*np.sqrt(t); S=X.T@X/n; w,V=np.linalg.eigh(S)
        rho.append((V[:k,:k]**2).sum()/k)
    print(p,k,lam,'n=',n,'A2=%.0f'%A2,'eig %.4f pred %.4f'%(np.mean(evs),1-A1/n),'rho %.4f pred %.4f'%(np.mean(rho),(1-A2/n)/(1-A1/n)))
