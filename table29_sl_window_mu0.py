# Table 29: mu_0 against window length, real Schmidt-Lipson spring-mass and pendulum.
# Needs the lawsample package (lawsample_pkg/src) and the archive file under sl/tds/data/schmidt-lipson-exp-data/.
import sys; sys.path.insert(0,'lawsample_pkg/src')
import numpy as np
from scipy.signal import savgol_filter
from lawsample.static_test import threshold_test
D="sl/tds/data/schmidt-lipson-exp-data/"
def run(name,cols,names,wins):
    d=np.loadtxt(D+name+".txt",comments="%"); x,v=d[:,2],d[:,3]
    X=np.c_[cols[0](x),cols[1](v)]
    sg=lambda a: savgol_filter(a,9,3)
    sd=np.array([np.std(X[:,0]-sg(X[:,0])),np.std(X[:,1]-sg(X[:,1]))])
    print(name,"rows",len(X),"noise sd",sd)
    for w in wins:
        mus=[]
        for a in range(0,len(X)-w+1,w):
            r=threshold_test(X[a:a+w],names,sd,law=None,degree=2,inject=(0.0,),m_grid=(),draws=1,reps=40,seed=1)
            mus.append(r[0]['mu0'])
        print(" window",w,"count",len(mus),"median mu0 %.2f"%np.median(mus))
run("real_linear_h_1",(lambda x:x,lambda v:v),"xv",(len(np.loadtxt(D+"real_linear_h_1.txt",comments="%")),200,100,50,30))
run("real_pend_h_1",(np.cos,lambda v:v),"cw",(len(np.loadtxt(D+"real_pend_h_1.txt",comments="%")),150,75,40,25))
