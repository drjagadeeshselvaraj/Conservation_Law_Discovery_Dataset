import numpy as np, json, sys
N=10; d=2*N
iu=np.triu_indices(d); p=len(iu[0]); k=N
a_,b_=iu
idx={(a,b):i for i,(a,b) in enumerate(zip(*iu))}
C=np.zeros((p,k))
for i in range(N):
    C[idx[(i,i)],i]=1; C[idx[(N+i,N+i)],i]=1
Qc,_=np.linalg.qr(C); PC=Qc@Qc.T
def rows(n,om,sx,sd,rg):
    r=np.sqrt(2*rg.exponential(1.0,(n,N))); th=rg.uniform(0,2*np.pi,(n,N))
    q=r*np.cos(th); pp=r*np.sin(th)
    x=np.hstack([q,pp]); xd=np.hstack([om*pp,-om*q])
    clean=xd[:,a_]*x[:,b_]+x[:,a_]*xd[:,b_]
    x=x+sx*rg.standard_normal(x.shape); xd=xd+sd*rg.standard_normal(xd.shape)
    return xd[:,a_]*x[:,b_]+x[:,a_]*xd[:,b_],clean
def pops(om,sx,sd,m=400000):
    rg=np.random.default_rng(5); S=np.zeros((p,p)); Nn=np.zeros((p,p)); Sc=np.zeros((p,p))
    for _ in range(m//50000):
        F,Cl=rows(50000,om,sx,sd,rg); S+=F.T@F; E=F-Cl; Nn+=E.T@E; Sc+=Cl.T@Cl
    return S/m,Nn/m,Sc/m
def al(Cm):
    Q,_=np.linalg.qr(Cm); return np.trace(PC@(Q@Q.T))/k
def Sn_of(n,om,sx,sd,rg):
    S=np.zeros((p,p)); left=n
    while left>0:
        c=min(50000,left); F,_c=rows(c,om,sx,sd,rg); S+=F.T@F; left-=c
    return S/n
scen=[("separated, noise 1",1+0.4*np.arange(N),0.1,0.2),
      ("separated, noise 2",1+0.4*np.arange(N),0.2,0.4),
      ("near-resonant pair",np.r_[1+0.4*np.arange(N-1),1+0.4*(N-2)+0.02],0.1,0.2)]
res=[]
for lab,om,sx,sd in scen:
    S,Nn,Sc=pops(om,sx,sd)
    ev,U=np.linalg.eigh(Nn); W=U@np.diag(ev**-0.5)@U.T
    w,V=np.linalg.eigh(W@S@W)
    ceil=al(W@V[:,:k]); a0=w[:k].mean(); l=w[k:]; t=l/(a0-l); A1=-t.sum(); A2=(t**2).sum()
    cl=np.linalg.eigvalsh(Sc)
    print(lab,"clean eig 11-12th",cl[k:k+2],"whitened null",w[:k].min().round(4),w[:k].max().round(4),"next",w[k].round(4),"ceil",round(ceil,3),"A1",round(A1),"A2",round(A2),"A2/p",round(A2/p,1),flush=True)
    rg=np.random.default_rng(99); out=[]
    fs=[0.4,0.7,0.9,1.1,1.5,2,4,8] if A2<3e4 else [0.01,0.03,0.1,0.3]
    for f in fs:
        n=int(max(f*A2,p+10)); n=min(n,400000); rn=[];rw=[]
        for _ in range(8):
            Sn=Sn_of(n,om,sx,sd,rg)
            vv,VV=np.linalg.eigh(Sn); rn.append(al(VV[:,:k]))
            vv,VV=np.linalg.eigh(W@Sn@W); rw.append(al(W@VV[:,:k]))
        pred=ceil*(max(0,(1-A2/n)/(1-A1/n)) if n>A2 else 0)
        out.append(dict(x=n/A2,n=n,naive=float(np.mean(rn)),whit=float(np.mean(rw)),sd=float(np.std(rw)),pred=float(pred)))
        print("   n=%d n/A2=%.3f naive=%.3f whit=%.3f±%.3f pred=%.3f"%(n,n/A2,out[-1]['naive'],out[-1]['whit'],out[-1]['sd'],pred),flush=True)
    res.append(dict(label=lab,A1=float(A1),A2=float(A2),ceil=float(ceil),rows=out))
    json.dump(res,open("osc3.json","w"))
