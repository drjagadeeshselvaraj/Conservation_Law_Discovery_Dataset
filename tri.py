import numpy as np, scipy.io as sio, scipy.linalg as sl, itertools, json
from scipy.signal import savgol_filter
D="/home/claude/dynamicslab/multiarm-pendulum/Datas/TriplePendulum/"
def build():
    spec=[]
    for i in range(3):
        for j in range(i,3):
            spec.append(("ww",i,j,None,f"w{i+1}w{j+1}"))
            if i!=j:
                spec.append(("ww",i,j,"c",f"w{i+1}w{j+1}cD{i+1}{j+1}")); spec.append(("ww",i,j,"s",f"w{i+1}w{j+1}sD{i+1}{j+1}"))
    for i in range(3): spec.append(("a",i,i,"c",f"cos{i+1}")); spec.append(("a",i,i,"s",f"sin{i+1}"))
    for i in range(3):
        for j in range(i+1,3): spec.append(("D",i,j,"c",f"cD{i+1}{j+1}")); spec.append(("D",i,j,"s",f"sD{i+1}{j+1}"))
    return spec
spec=build(); names=[s_[4] for s_ in spec]; P=len(names)
def feat_all(T,W,A):
    cols=[];dcols=[]
    for kind,i,j,g,_ in spec:
        if kind=="ww":
            D=T[i]-T[j]; dD=W[i]-W[j]
            base=W[i]*W[j]; dbase=A[i]*W[j]+W[i]*A[j]
            if g is None: cols.append(base); dcols.append(dbase)
            elif g=="c": cols.append(base*np.cos(D)); dcols.append(dbase*np.cos(D)-base*np.sin(D)*dD)
            else: cols.append(base*np.sin(D)); dcols.append(dbase*np.sin(D)+base*np.cos(D)*dD)
        elif kind=="a":
            if g=="c": cols.append(np.cos(T[i])); dcols.append(-np.sin(T[i])*W[i])
            else: cols.append(np.sin(T[i])); dcols.append(np.cos(T[i])*W[i])
        else:
            D=T[i]-T[j]; dD=W[i]-W[j]
            if g=="c": cols.append(np.cos(D)); dcols.append(-np.sin(D)*dD)
            else: cols.append(np.sin(D)); dcols.append(np.cos(D)*dD)
    return np.stack(dcols,1),np.stack(cols,1)
def load(i):
    m=sio.loadmat(D+f"TripleDataFreeSwing_{i}_Dt_0_0001.mat"); g=lambda k:m[k].ravel()
    return [g("Theta1"),g("Theta2"),g("Theta3")],[g("dTheta1"),g("dTheta2"),g("dTheta3")],float(m["dt"].ravel()[0])
def prep(i,win=301,step=200):
    T,W,dt=load(i)
    Ts=[savgol_filter(t,win,3) for t in T]; Ws=[savgol_filter(t,win,3,deriv=1,delta=dt) for t in T]; As=[savgol_filter(t,win,3,deriv=2,delta=dt) for t in T]
    st=np.mean([np.std(t-s) for t,s in zip(T,Ts)]); sw=np.mean([np.std(w-s) for w,s in zip(W,Ws)])
    sa=np.mean([np.std(savgol_filter(w,win,3,deriv=1,delta=dt)-a) for w,a in zip(W,As)])
    idx=np.arange(win,len(T[0])-win,step)
    return dict(T=[t[idx] for t in Ts],W=[w[idx] for w in Ws],A=[a[idx] for a in As],st=st,sw=sw,sa=sa,dt=dt)
def feats(T,W,A): return feat_all(T,W,A)
def FN(d,rg,reps=15):
    n=len(d["T"][0]); F0,Pm=feats(d["T"],d["W"],d["A"]); N=0
    z=lambda s:s*rg.standard_normal(n)
    for _ in range(reps):
        F=feats([t+z(d["st"]) for t in d["T"]],[w+z(d["sw"]) for w in d["W"]],[a+z(d["sa"]) for a in d["A"]])[0]
        E=F-F0; N=N+E.T@E/n
    return F0,N/reps,Pm
