# Table 28 (Boyle), REBUILT from the method in the text, not the original script.
# Needs lawsample_pkg/src on the path and boyle.csv (25 rows: volume, pressure).
import sys; sys.path.insert(0,'lawsample_pkg/src')
import numpy as np
from lawsample.static_test import threshold_test
d=np.loadtxt('boyle.csv',delimiter=',',skiprows=1); V,P=d[:,0],d[:,1]
print('rows',len(d),'mean PV %.0f sd %.1f'%((P*V).mean(),(P*V).std()))
X=np.c_[V,P]
for rel in (0.002,0.0035,0.005):
    sd=np.c_[np.zeros(len(P)),rel*P]
    r=threshold_test(X,"VP",sd,law={"VP":1.0,"1":-1408.0},degree=2,inject=(0.0,),m_grid=(8,25),draws=40,reps=40,seed=1)[0]
    print('noise %.4f mu0 %.2f A2 %.1f'%(rel,r['mu0'],r['A2']),[(m,round(a,2),round(pl,2)) for m,a,f,pl in r['rows']])
