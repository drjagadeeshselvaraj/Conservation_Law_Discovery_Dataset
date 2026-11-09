import numpy as np,sys
kind=sys.argv[1]; rg=np.random.default_rng(11)
def lam(m):
    if kind=="equal": return np.full(m,1.0)
    if kind=="graded":return np.linspace(0.5,3.0,m)
def run(p,ratio,reps):
    l=lam(p-1);A2=np.sum((1+1/l)**2);n=int(round(ratio*A2));sd=np.r_[1.0,np.sqrt(1+l)];low=[]
    for _ in range(reps):
        X=rg.standard_normal((n,p))*sd
        w,U=np.linalg.eigh(X.T@X/n); low.append(U[0,0]**2)
    return np.array(low),A2/n-1
for ratio in [0.3,0.4,0.5,0.6,0.7,0.8]:
    row=[]
    for p,reps in [(250,120),(500,60),(1000,30),(2000,12)]:
        low,d=run(p,ratio,reps); row.append((p,low.mean()*p,low.std()/np.sqrt(reps)*p))
    print(kind,ratio,"delta=%.3f"%d," ".join("p%d:p*low=%.2f±%.2f"%r for r in row),flush=True)
