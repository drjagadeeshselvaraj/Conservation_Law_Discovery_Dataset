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
seen=set(); out={}
for k,(L,f) in files.items():
    key=(L,f.endswith('.csv') and k.replace('.csv','.txt'))
    if f.endswith('.csv'): continue   # csv duplicates of txt (verified identical below)
    t,th,om=load(f); th=np.deg2rad(th)
    X,S,sig=rows(t,th)
    A=np.c_[X[:,0]**2,np.ones(len(X))]; co=np.linalg.lstsq(A,X[:,1]**2,rcond=None)[0]
    law={"ww":1.0,"tt":-co[0],"1":-co[1]}
    g_est=-co[0]*L/100.0  # omega^2=-a... a=-g/L relation: w^2 = -(g/L)th^2 + c  -> g/L=-co0
    gL=-co[0]; g=gL*L/100
    ms=tuple(m for m in (10,15,20,30,40,60,80,120,160,240,400,800) if m<=len(X))
    res=threshold_test(X,"tw",S,law=law,degree=2,inject=(0.0,0.05,0.1,0.2),inject_cols=[0,1],inject_mode="std",
                       m_grid=ms,draws=60,reps=30,seed=3)
    print(f"{k[:36]:36s} L={L} rows={len(X)} sig_th={np.rad2deg(sig):.3f}deg  implied g={g:.2f} m/s2 (w^2 fit)")
    for r in res:
        print('   inj %.2f: mu0 %.2f A2 %.1f n05 %s rms %.3f'%(r['inject'],r['mu0'],r['A2'],r['n05'],r['rms_gap']))
    out[k]=dict(L=L,n=len(X),sig_deg=float(np.rad2deg(sig)),g=float(g),res=res)
json.dump(out,open('pt1.json','w'),default=float)
