import numpy as np
from scipy.integrate import solve_ivp
c=0.002
m=np.array([1.0,0.8,0.6]); l=np.array([1.0,0.9,0.7]); g=9.81; 
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

def run(a,cc,rg,Tend=60.0,dt=1e-4):
    global c
    c=cc
    t=np.arange(0,Tend,dt); y0=np.array(list(a)+[0.3*rg.standard_normal() for _ in range(3)])
    sol=solve_ivp(f,(0,Tend),y0,t_eval=t,rtol=1e-10,atol=1e-12,method="DOP853"); return sol.y,t,dt
