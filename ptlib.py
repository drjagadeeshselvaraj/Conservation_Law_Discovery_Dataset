import sys; R='/tmp/claude-0/-home-claude/1bdb3634-9951-583d-9a91-beca2986667c/scratchpad/'
sys.path.insert(0,R+'pt_pt'); sys.path.insert(0,R+'lawsample_pkg/src')
from load import *; import json
from lawsample.static_test import threshold_test
W=9; h=W//2
def rows(t,th):
    X=[];S=[]
    # residual sd from overlapping local quadratic
    res=[]
    for i in range(h,len(t)-h):
        s=slice(i-h,i+h+1); c=np.polyfit(t[s]-t[i],th[s],2); res.append(th[i]-c[2])
    sig=np.std(res)*np.sqrt(W/(W-3))   # unbias for 3 fitted params
    for i in range(h,len(t)-h,W):
        s=slice(i-h,i+h+1); tt=t[s]-t[i]
        if tt[-1]-tt[0]>0.4: continue
        D=np.vander(tt,3,increasing=True); cov=np.linalg.inv(D.T@D)
        c=np.linalg.lstsq(D,th[s],rcond=None)[0]
        X.append([c[0],c[1]]); S.append([sig*np.sqrt(cov[0,0]),sig*np.sqrt(cov[1,1])])
    return np.array(X),np.array(S),sig
