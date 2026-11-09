import numpy as np, pandas as pd, itertools, json
from scipy.linalg import eigh
rg=np.random.default_rng(0)
d=pd.read_csv('g.csv'); X=d[['n','T','V','P']].values
names=['1']+['n','T','V','P']
combos=[()]+[(i,) for i in range(4)]+[c for c in itertools.combinations_with_replacement(range(4),2)]
nm=['n','T','V','P']
lab=['1']+[nm[i] for i in range(4)]+[nm[a]+nm[b] for a,b in itertools.combinations_with_replacement(range(4),2)]
def feats(X):
    cols=[np.ones(len(X))]+[X[:,i] for i in range(4)]+[X[:,a]*X[:,b] for a,b in itertools.combinations_with_replacement(range(4),2)]
    return np.stack(cols,1)
F=feats(X); p=F.shape[1]
cstar=np.zeros(p); cstar[lab.index('VP')]=1; cstar[lab.index('nT')]=-1
sigP=0.01*0.98   # 1% of the clean target's std (P std ~0.98)
# clean-law residual size (should be ~ noise): |F c*| ~ V*eps
r=F@cstar; print('p',p,'std of F c*',r.std(),' expected ~ sigP*mean(V)=',sigP*X[:,2].mean())
# feature-noise covariance by resampling P noise at the observed rows
def noise_cov(reps=20):
    D=[]
    for _ in range(reps):
        Xn=X.copy(); Xn[:,3]+=sigP*rg.standard_normal(len(X)); D.append(feats(Xn)-F)
    D=np.vstack(D); return D.T@D/len(D)
N=noise_cov()
scc=np.sqrt((F**2).mean(0)); Fs=F/scc; Ns=N/np.outer(scc,scc)+1e-14*np.eye(p)
S=Fs.T@Fs/len(Fs)
mu,U=eigh(S,Ns)
print('mu (smallest 5):',mu[:5].round(3),' largest',mu[-3:].round(1))
lam=mu[1:]-1; print('non-positive lam count',int((lam<=0).sum())); lam=lam[lam>0]
A1=np.sum(1+1/lam); A2=np.sum((1+1/lam)**2); print('A1 %.0f A2 %.0f'%(A1,A2))
cs=cstar*scc
ca=lambda a:(a@Ns@cs)**2/((a@Ns@a)*(cs@Ns@cs))
print('alignment of full-data whitened eigvec with true law:',round(float(ca(U[:,0])),4))
def stlsq_null(Fm,tau=0.1,iters=6):
    best=None
    for j in range(Fm.shape[1]):
        idx=[q for q in range(Fm.shape[1]) if q!=j]; A=Fm[:,idx]; b=-Fm[:,j]
        c=np.linalg.lstsq(A,b,rcond=None)[0]
        for _ in range(iters):
            sm=np.abs(c)<tau; c[sm]=0; big=~sm
            if big.sum()==0: break
            c[big]=np.linalg.lstsq(A[:,big],b,rcond=None)[0]
        v=np.zeros(Fm.shape[1]); v[idx]=c; v[j]=1.0
        r=np.linalg.norm(Fm@v)/np.linalg.norm(v)
        if best is None or r<best[0]: best=(r,v/np.linalg.norm(v))
    return best[1]
rows=[]
for m in (20,30,50,100,200,400,800,1600,3200):
    aw=[];ap=[];asd=[]
    for _ in range(60):
        ii=rg.choice(len(X),m,replace=False); Sn=Fs[ii].T@Fs[ii]/m
        w,Un=eigh(Sn,Ns); aw.append(ca(Un[:,0]))
        w2,U2=np.linalg.eigh(Sn); ap.append(ca(U2[:,0]))
        if m<=800: asd.append(ca(stlsq_null(Fs[ii])))
    pred=max(0,(1-A2/m)/(1-A1/m)) if m>A2 else 0.0
    rows.append((m,round(float(np.mean(aw)),3),round(pred,3),round(float(np.mean(ap)),3),round(float(np.mean(asd)),3) if asd else None))
    print(rows[-1],flush=True)
json.dump(dict(A1=A1,A2=A2,mu=mu[:5].tolist(),rows=rows),open('gas.json','w'))
