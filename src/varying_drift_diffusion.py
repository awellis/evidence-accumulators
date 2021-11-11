import numpy as np
from numba import njit, prange
from tensorflow.python.keras.utils.np_utils import to_categorical
from accumulators import *

@njit
def var_dm_simulator(theta, n_obs, movement_profile, s=1.0, dt=0.001, max_iter=1e4):
    # parameters
    a     = theta[0] # boundary separation
    ndt   = theta[1] # non-decision time
    bias  = theta[2] # a priori bias
    kappa = theta[3] # sensitivity to movement profile

    # initialize data
    resp = np.empty(n_obs, dtype=np.int32)
    rt   = np.empty(n_obs, dtype=np.float32)

    # iterate over trials
    for n in range(n_obs):

        drift = kappa * movement_profile[n]**2
        rt[n], resp[n] = varying_evidence_accumulation(drift, a, ndt, bias, kappa, s, dt, max_iter)

    return rt, resp

def var_dm_priors(n_sim=1):

    a     = np.random.uniform(0.5, 3.0, size=n_sim)
    ndt   = np.random.uniform(0.1, 0.5, size=n_sim)
    bias  = np.random.normal(0.5, 0.05, size=n_sim)
    kappa = np.random.uniform(0.0, 3.0, size=n_sim)

    theta = np.array([a, ndt, bias, kappa]).T

    return theta

def var_dm_batch_simulator(n_sim, n_obs):
    prior_samples = var_dm_priors(n_sim)
    movement_profile = None

    # simulate
    sim_data = var_dm_batch_simulator_wrap(prior_samples, movement_profile, n_sim, n_obs)

    # data prep
    one_hot_encoded_resp = to_categorical(sim_data[:, :, 1])
    return prior_samples, np.c_[np.expand_dims(sim_data[:, :, 0], axis=2), one_hot_encoded_resp]

# wrapper function for faster simulation
@njit(parallel=True)
def var_dm_batch_simulator_wrap(prior_samples, movement_profile, n_sim, n_obs):
    sim_data = np.zeros((n_sim, n_obs, 2), dtype=np.float32)
    # iterate over simulations
    for sim in prange(n_sim):
        rt, resp = var_dm_simulator(prior_samples[sim], n_obs, movement_profile)
        sim_data[sim] = np.vstack((rt, resp)).T
    return sim_data