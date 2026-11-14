import numpy as np

def make_diagonal(x):
    diag = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        diag[i][i] = x[i]

    return diag