import sys; sys.path.insert(0,'../real2nd')
from tuned import sindy_tuned
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
import sys
levels=[0.0,0.1,0.3]
sigP0=0.01*0.98

out={}
for fr in levels:
    sigP=np.sqrt(sigP0**2+(fr*0.98)**2)
    Xb=X.copy()
    Xb[:,3]=X[:,3]+(fr*0.98)*rg.standard_normal(len(X))    # injected extra noise on P
    Fb=feats(Xb)
    def noise_cov(reps=20):
        D=[]
        for _ in range(reps):
            Xn=Xb.copy(); Xn[:,3]+=sigP*rg.standard_normal(len(X)); D.append(feats(Xn)-Fb)
        D=np.vstack(D); return D.T@D/len(D)
    N=noise_cov()
    scc=np.sqrt((Fb**2).mean(0)); Fs=Fb/scc; Ns=N/np.outer(scc,scc)+1e-14*np.eye(p)
    S=Fs.T@Fs/len(Fs); mu,U=eigh(S,Ns)
    lam=mu[1:]-1; nneg=int((lam<=0).sum()); lam=lam[lam>0]
    A1=np.sum(1+1/lam); A2=np.sum((1+1/lam)**2)
    cs=cstar*scc
    ca=lambda a:(a@Ns@cs)**2/((a@Ns@a)*(cs@Ns@cs))
    print(f"inj {fr}: mu0 {mu[0]:.2f} mu1 {mu[1]:.1f} A1 {A1:.0f} A2 {A2:.0f} nneg {nneg}",flush=True)
    rows=[]
    for m in (30,60,100,200,400,800,1600):
        aw=[];ap=[];asd=[]
        for _ in range(40):
            ii=rg.choice(len(X),m,replace=False); Sn=Fs[ii].T@Fs[ii]/m
            w,Un=eigh(Sn,Ns); aw.append(ca(Un[:,0])); w2,U2=np.linalg.eigh(Sn); ap.append(ca(U2[:,0])); v,_=sindy_tuned(Fs[ii],rg=rg); asd.append(ca(v))
        pred=max(0,(1-A2/m)/(1-A1/m)) if m>A2 else 0.0
        rows.append((m,round(float(np.mean(aw)),3),round(pred,3),round(float(np.mean(ap)),3),round(float(np.mean(asd)),3)))
    print('  (m, whitened, formula, plain, SINDy tuned):',rows,flush=True)
    meas=next((m for m,a,_,_,_ in rows if a>=0.5),None)
    gap=float(np.sqrt(np.mean([(a-b)**2 for _,a,b,_,_ in rows])))
    print('  measured n0.5',meas,' formula A2 %.0f'%A2,' rms gap %.3f'%gap,flush=True)
    out[fr]=dict(A1=A1,A2=A2,mu0=mu[0],mu1=mu[1],rows=rows,meas=meas,gap=gap)
json.dump(out,open('gas_tuned.json','w'),default=float)
