import numpy as np,sys
rg=np.random.default_rng(int(sys.argv[2]))
kind=sys.argv[1]
def lam(kind,m):
    if kind=="equal": return np.full(m,1.0)
    if kind=="two":   return np.r_[np.full(m//2,0.5),np.full(m-m//2,2.0)]
    if kind=="graded":return np.linspace(0.5,3.0,m)
def run(p,ratio,reps):
    l=lam(kind,p-1);A1=np.sum(1+1/l);A2=np.sum((1+1/l)**2);n=int(round(ratio*A2))
    sd=np.r_[1.0,np.sqrt(1+l)];mx=[];low=[]
    for _ in range(reps):
        X=(rg.standard_normal((n,p))*sd).astype(np.float64)
        w,U=np.linalg.eigh(X.T@X/n); o=U[0]**2
        mx.append(o.max());low.append(o[0])
    pred=max(0,(1-A2/n)/(1-A1/n)) if n>A2 else 0
    return np.mean(mx),np.std(mx)/np.sqrt(reps),np.mean(low),pred
cfg=[(250,40),(500,20),(1000,10),(2000,5)]
for ratio in [0.5,0.7,0.8,0.9,0.95,1.1,1.3,1.6]:
    out=[]
    for p,r in cfg:
        m,se,lo,pr=run(p,ratio,r); out.append((p,m,se,lo,pr))
    print(kind,ratio,"pred=%.3f"%out[0][4],"|"," ".join("p%d:max=%.4f±%.4f,low=%.3f"%(p,m,se,lo) for p,m,se,lo,_ in out),flush=True)
