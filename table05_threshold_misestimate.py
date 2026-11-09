# Table 5: ratio of predicted to true threshold when the noise is estimated f times too high.
# Pure formula: (A2_est/A2_true) = ((1 + f/lam0)/(1 + 1/lam0))**2
for lam0 in (0.5, 1, 3, 10):
    print(lam0, [round(((1 + f/lam0)/(1 + 1/lam0))**2, 2) for f in (0.25, 0.5, 2, 4, 13)])
