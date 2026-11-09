import sys, numpy as np, tb
s0=int(sys.argv[1]); X=tb.orbits(norb=100,seed0=s0); np.save(f"X_{s0}.npy",X); print(X.shape)
