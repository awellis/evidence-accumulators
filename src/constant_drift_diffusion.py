import numpy as np
from numba import njit, prange
from tensorflow.python.keras.utils.np_utils import to_categorical
from accumulators import *
from motion_simulation import *

@njit
def const_dm_simulator(theta, n_obs, amplitude, s=0.1, dt=0.001, max_iter=1e4):
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
        drift = kappa * amplitude[n]
        rt[n], resp[n] = const_evidence_accumulation(drift, a, ndt, bias, s, dt, max_iter)

    return rt, resp

def var_dm_priors(n_sim=1):

    a     = np.random.uniform(0.5, 3.0, size=n_sim)
    ndt   = np.random.uniform(0.1, 0.5, size=n_sim)
    bias  = np.random.uniform(0.2, 0.8, size=n_sim)
    kappa = np.random.uniform(0.0, 10.0, size=n_sim)

    theta = np.array([a, ndt, bias, kappa]).T

    return theta

def const_dm_batch_simulator(n_sim, n_obs):
    prior_samples = var_dm_priors(n_sim)

    unique_motions = np.array([-0.725, -0.675, -0.625, -0.575, -0.525, 0.525,  0.575,  0.625,  0.675,  0.725], dtype=np.float32)
    amplitude = np.repeat(unique_motions, 10)
    condition = to_categorical(pd.factorize(amplitude)[0])

    # simulate
    sim_data = const_dm_batch_simulator_wrap(prior_samples, amplitude, condition, n_sim, n_obs)

    # data prep
    one_hot_encoded_resp = to_categorical(sim_data[:, :, 1])
    return prior_samples, np.c_[np.expand_dims(sim_data[:, :, 0], axis=2), one_hot_encoded_resp, sim_data[:, :, 3:]]

# wrapper function for faster simulation
@njit
def const_dm_batch_simulator_wrap(prior_samples, amplitude, condition, n_sim, n_obs):
    sim_data = np.zeros((n_sim, n_obs, condition.shape[1] + 2), dtype=np.float32)
    # iterate over simulations
    for sim in range(n_sim):
        # shuffle motion profile together with condition
        shuffler = np.random.permutation(len(amplitude))
        amplitude = amplitude[shuffler]
        condition = condition[shuffler, :]

        # simulate trials
        rt, resp = const_dm_simulator(prior_samples[sim], n_obs, amplitude)
        sim_data[sim] = np.hstack((np.expand_dims(rt, axis=1), np.expand_dims(resp, axis=1), condition[0:n_obs, :]))

    return sim_data
