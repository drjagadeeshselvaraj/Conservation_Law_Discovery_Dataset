import numpy as np
from scipy.integrate import solve_ivp
m=np.array([1.0,0.8,0.6]); l=np.array([1.0,0.9,0.7]); g=9.81; c=0.01
Msum=np.array([m[i:].sum() for i in range(3)])
coef=np.array([[Msum[max(i,j)]*l[i]*l[j] for j in range(3)] for i in range(3)])
def f(t,y):
    th=y[:3]; w=y[3:]
    D=th[:,None]-th[None,:]
    M=coef*np.cos(D)
    rhs=-(coef*np.sin(D))@(w**2)-g*l*Msum*np.sin(th)-c*w
    return np.concatenate([w,np.linalg.solve(M,rhs)])
def energy(y):
    th=y[:3];w=y[3:];D=th[:,None]-th[None,:]
    return 0.5*w@(coef*np.cos(D))@w-g*np.sum(l*Msum*np.cos(th))
rg=np.random.default_rng(7)
dt=1e-4; Tend=60.0; t=np.arange(0,Tend,dt)
for i,(a) in enumerate([(1.2,0.9,0.5),(0.8,-1.1,1.4),(1.5,0.4,-0.9)],1):
    y0=np.array(list(a)+[0.3*rg.standard_normal() for _ in range(3)])
    sol=solve_ivp(f,(0,Tend),y0,t_eval=t,rtol=1e-10,atol=1e-12,method="DOP853")
    Y=sol.y; e=[energy(Y[:,k]) for k in (0,len(t)//2,len(t)-1)]
    T=[Y[k]+3e-5*rg.standard_normal(len(t)) for k in range(3)]
    np.savez(f"simB_run{i}.npz",T1=T[0],T2=T[1],T3=T[2])
    print("run",i,"energy",e,flush=True)
