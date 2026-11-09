import numpy as np, json, sys
exec(open("osc3.py").read().split("scen=[")[0])
def stage(Sn,Nn,sel):
    Ns=Nn[np.ix_(sel,sel)]; Ss=Sn[np.ix_(sel,sel)]
    e,U=np.linalg.eigh(Ns); Ws=U@np.diag(e**-0.5)@U.T
    v,V=np.linalg.eigh(Ws@Ss@Ws); Cs=Ws@V[:,:k]
    Q,_=np.linalg.qr(Cs); return Cs,(Q**2).sum(1)
def iterative(Sn,Nn,sched):
    sel=np.arange(p)
    for m in sched:
        Cs,lev=stage(Sn,Nn,sel)
        if m<len(sel): sel=sel[np.argsort(-lev)[:m]]
    Cs,lev=stage(Sn,Nn,sel); full=np.zeros((p,k)); full[sel,:]=Cs; return full
def implicit_sindy(S,thr=0.1,iters=5):
    cands=[];score=[]
    for j in range(p):
        idx=np.array([i for i in range(p) if i!=j]); A=S[np.ix_(idx,idx)]; b=-S[idx,j]
        try: c=np.linalg.solve(A+1e-12*np.eye(p-1),b)
        except np.linalg.LinAlgError: continue
        for _ in range(iters):
            m=np.abs(c)>=thr*max(1e-12,np.abs(c).max())
            if m.sum()==0: break
            c2=np.zeros(p-1); 
            try: c2[m]=np.linalg.solve(A[np.ix_(m,m)]+1e-12*np.eye(m.sum()),b[m])
            except np.linalg.LinAlgError: break
            c=c2
        v=np.zeros(p); v[j]=1; v[idx]=c
        score.append(v@S@v/(v@v)); cands.append(v/np.linalg.norm(v))
    order=np.argsort(score)[:2*k]; M=np.array([cands[i] for i in order]).T
    U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
def sched(mf): return [s for s in [140,90,60,40] if s>mf]+[mf]
out={"heldout":[]}
rgs=np.random.default_rng(2024)
def draw_omega():
    while True:
        w=np.sort(rgs.uniform(0.8,4.0,N))
        if np.min(np.diff(w))>=0.25: return w
oms=[draw_omega() for _ in range(5)]
noises=[("noise 1",0.1,0.2),("noise 2",0.2,0.4)]
ns=[300,600,1500,3000]; REPS=5
for si,om in enumerate(oms):
    for lab,sx,sd in noises:
        S,Nn,Sc=pops(om,sx,sd,m=200000)
        ev,U=np.linalg.eigh(Nn); W=U@np.diag(ev**-0.5)@U.T
        rg=np.random.default_rng(100+si)
        for n in ns:
            r={k_:[] for k_ in ["plain_whitened","sindy_raw","sindy_noisecorr","iter_m20","iter_m24","iter_m30","iter_m40"]}
            for _ in range(REPS):
                Sn=Sn_of(n,om,sx,sd,rg)
                vv,VV=np.linalg.eigh(W@Sn@W); r["plain_whitened"].append(al(W@VV[:,:k]))
                r["sindy_raw"].append(al(implicit_sindy(Sn)))
                r["sindy_noisecorr"].append(al(implicit_sindy(Sn-Nn)))
                for mf in [20,24,30,40]: r["iter_m%d"%mf].append(al(iterative(Sn,Nn,sched(mf))))
            row=dict(set=si,noise=lab,n=n,**{k_:float(np.mean(v)) for k_,v in r.items()})
            out["heldout"].append(row); print(row,flush=True)
            json.dump(out,open("exp4.json","w"))
