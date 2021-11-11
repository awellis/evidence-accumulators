import numpy as np
from numba import njit, prange

@njit
def varying_evidence_accumulation(drift, a, ndt, bias, s=1.0, dt=0.001, max_iter=1e4):
    # constant for diffusion process
    c = np.sqrt(dt * s)
    # starting point
    x = a * bias

    # accumulation process
    n_iter = 0
    while x < a and x > 0 and n_iter < max_iter:
        x += drift[n_iter]*dt + c*np.random.randn()
        n_iter += 1
    
    if n_iter < max_iter:
        rt = n_iter*dt + ndt
        resp = 1 if x>0 else 0
    else:
        rt = 0
        resp = 0

    return rt, resp

@njit
def const_evidence_accumulation(drift, a, ndt, bias, s=1.0, dt=0.001, max_iter=1e4):
    # constant for diffusion process
    c = np.sqrt(dt * s)
    # starting point
    x = a * bias

    # accumulation process
    n_iter = 0
    while x < a and x > 0 and n_iter < max_iter:
        x += drift*dt + c*np.random.randn()
        n_iter += 1
    
    if n_iter < max_iter:
        rt = n_iter*dt + ndt
        resp = 1 if x>0 else 0
    else:
        rt = 0
        resp = 0

    return rt, resp