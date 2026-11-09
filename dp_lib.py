import numpy as np
from scipy.signal import savgol_filter
from scipy.linalg import eigh
def load(tr):
    m=[np.load(f'vid/Video_Tracking_Data/Trial{tr}/DPmean_data_RB{rb}.npy') for rb in (0,1)]
    s=[np.load(f'vid/Video_Tracking_Data/Trial{tr}/DPstd_data_RB{rb}.npy') for rb in (0,1)]
    t=m[0][0]; th=np.deg2rad(np.stack([m[0][1],m[1][1]])); sd=np.deg2rad(np.stack([s[0][1],s[1][1]]))
    return t,th,sd
names=['w1w1','w2w2','w1w2','w1w1cD','w2w2cD','w1w2cD','w1w1sD','w2w2sD','w1w2sD','cos1','sin1','cos2','sin2','cosD','sinD']
def feat(T,W,A):
    t1,t2=T; w1,w2=W; a1,a2=A; D=t1-t2; dD=w1-w2
    base=[(w1*w1,2*w1*a1),(w2*w2,2*w2*a2),(w1*w2,a1*w2+w1*a2)]
    cols=[];d=[]
    for b,db in base: cols.append(b); d.append(db)
    for b,db in base: cols.append(b*np.cos(D)); d.append(db*np.cos(D)-b*np.sin(D)*dD)
    for b,db in base: cols.append(b*np.sin(D)); d.append(db*np.sin(D)+b*np.cos(D)*dD)
    cols+= [np.cos(t1),np.sin(t1),np.cos(t2),np.sin(t2),np.cos(D),np.sin(D)]
    d+= [-np.sin(t1)*w1,np.cos(t1)*w1,-np.sin(t2)*w2,np.cos(t2)*w2,-np.sin(D)*dD,np.cos(D)*dD]
    return np.stack(d,1),np.stack(cols,1)
def smooth(th,dt,win,order=3):
    return [savgol_filter(th,win,order,deriv=k,delta=dt,axis=-1) for k in (0,1,2)]
