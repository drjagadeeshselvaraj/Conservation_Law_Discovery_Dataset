# Table 24 (Gaia DR3 parallax x distance), REBUILT from the method in the text (original script not saved).
# Data: gaia_parallax_distance.csv (source_id, parallax, parallax_error, distance_gspphot, _lower, _upper).
# Library: monomials of degree <= 2 in (w, d/1000), p = 6; law e_wd - e_1, k = 1.
# Known noise: reported parallax error; half the reported distance interval (per-row).
import sys; sys.path.insert(0,'lawsample_pkg/src')
import numpy as np, pandas as pd
from lawsample.static_test import threshold_test
d=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'gaia2/gaia_parallax_distance.csv').dropna()
w=d.parallax.values; D=d.distance_gspphot.values/1000.0
sw=d.parallax_error.values; sD=0.5*(d.distance_gspphot_upper.values-d.distance_gspphot_lower.values)/1000.0
prod=w*D
def run(mask,label):
    X=np.c_[w[mask],D[mask]]; sd=np.c_[sw[mask],sD[mask]]
    r=threshold_test(X,"wd",sd,law={"wd":1.0,"1":-1.0},degree=2,inject=(0.0,),m_grid=(),draws=1,reps=40,seed=1)[0]
    print('%-28s n=%4d median prod %.2f  mu0 %.2f  A2 %.1f'%(label,mask.sum(),np.median(prod[mask]),r['mu0'],r['A2']))
run(np.ones(len(w),bool),'All')
run(np.abs(prod-1)<0.3,'|prod-1|<0.3')
run(w>4,'parallax > 4 mas')
print('robust scatter (1.4826 MAD) %.3f, sd %.3f'%(1.4826*np.median(np.abs(prod-np.median(prod))),prod.std()))
