import numpy as np, glob, re, os
A='/mnt/user-data/attachments/1bdb3634-9951-583d-9a91-beca2986667c/'
def load(f):
    rows=[]
    for ln in open(f,encoding='utf-8-sig'):
        p=ln.replace('"','').replace(',','.').replace(';',' ').split()
        try: v=[float(x) for x in p[:3]]
        except: continue
        if len(v)==3: rows.append(v)
    a=np.array(rows); return a[:,0],a[:,1],a[:,2]
files={}
for f in sorted(glob.glob(A+'*.txt')+glob.glob(A+'*.csv')):
    b=os.path.basename(f)
    if not re.search(r'cm|L_45',b) : continue
    m=re.search(r'(\d+\.?\d*)(?:_v\d)?cm',b) or re.search(r'L_(\d+\.\d)cm',b)
    L=float(m.group(1)); files[b[9:]]=(L,f)
