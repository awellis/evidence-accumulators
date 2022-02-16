import numpy as np
from numba import njit

@njit
def linear_collapsing_bound(a, c, max_iter=1e4):
    t = np.arange(max_iter) * 0.001

    a_upper = np.maximum(np.minimum(a - c*t, a), a/2)
    a_lower = np.maximum(np.minimum(a*(t / (t + c)), a/2), 0)

    return a_lower, a_upper


@njit
def not_linear_collapsing_bound(a, c, max_iter=1e4):
    t = np.arange(max_iter) * 0.001

    a_upper = np.maximum(np.minimum(a - a*(t / (t + c)), a), a/2)
    a_lower = np.maximum(np.minimum(a*(t / (t + c)), a/2), 0)

    return a_lower, a_upper


@njit
def exponential_collapsing_bound(a, tau, max_iter=1e4):
    """
    Starting point equals zero now.
    """
    t = np.arange(max_iter) * 0.001

    a_upper = (a/2) * np.exp(-tau*t)
    a_lower = -(a/2) * np.exp(-tau*t)

    return a_lower, a_upper


@njit
def weibull_collapsing_bound(a_0, a_asymp, scale, k, max_iter=1e4):
    """
    k=3 --> late collapse
    """
    t = np.arange(max_iter) * 0.001

    a_lower = (1 - np.exp(-(t/scale)**k)) * ((a_0/2) - a_asymp)
    a_upper = a_0 - (1 - np.exp(-(t/scale)**k)) * ((a_0/2) - a_asymp)

    return a_lower, a_upper
