import numpy as np
from numba import njit, prange
from tensorflow.python.keras.utils.np_utils import to_categorical
from accumulators import *
from motion_simulation import *
from pathlib import Path

@njit
def var_dm_simulator(theta, n_obs, motion_profile, s=1.0, dt=0.001, max_iter=1e4):
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
        drift = kappa * motion_profile[n]
        rt[n], resp[n] = varying_evidence_accumulation(drift, a, ndt, bias, s, dt, max_iter)

    return rt, resp

def var_dm_priors(n_sim=1):

    a     = np.random.uniform(0.5, 3.0, size=n_sim)
    ndt   = np.random.uniform(0.1, 0.5, size=n_sim)
    bias  = np.random.uniform(0.2, 0.8, size=n_sim)
    kappa = np.random.uniform(0.0, 5.0, size=n_sim)

    theta = np.array([a, ndt, bias, kappa]).T

    return theta

def var_dm_batch_simulator(n_sim, n_obs):
    prior_samples = var_dm_priors(n_sim)

    # create an motion experiment with fixed design (temporarily)
    # maybe I could just give the acceleration peak intensity to BayesFlow instead of 
    # one-hot-encoded condition 2darray.
    # n_unique_motions = 5
    # motion_dur = 2
    # motion_profile, condition = motion_experiment(n_obs, n_unique_motions, motion_dur)
    
    # get exact motion profile used in the data
    directory = str(Path().absolute())
    path = str(Path(directory).parents[1]) + '/evidence-accumulators/data/single_sub_data.csv'
    # data = np.loadtxt(open(path, 'rb'), delimiter=",", skiprows=1)
    data = pd.read_csv(path)
    idx = np.where((data["condition"] == 1) & (data["instruction"] == 1))
    data = data.loc[idx]
    amplitude = data["motion"]
    frequency = data["motion_duration"][0]
    motion_dur = data["motion_duration"][0]

    motion_set, condition = motion_experiment_manual(motion_dur, amplitude, frequency)

    # simulate
    sim_data = var_dm_batch_simulator_wrap(prior_samples, motion_set, condition, n_sim, n_obs)

    # data prep
    one_hot_encoded_resp = to_categorical(sim_data[:, :, 1])
    return prior_samples, np.c_[np.expand_dims(sim_data[:, :, 0], axis=2), one_hot_encoded_resp, sim_data[:, :, 3:]]

# wrapper function for faster simulation
@njit(parallel=True)
def var_dm_batch_simulator_wrap(prior_samples, motion_profile, condition, n_sim, n_obs):
    sim_data = np.zeros((n_sim, n_obs, condition.shape[1] + 2), dtype=np.float32)
    # iterate over simulations
    for sim in prange(n_sim):
        rt, resp = var_dm_simulator(prior_samples[sim], n_obs, motion_profile)
        # sim_data[sim] = np.c_((rt, resp, condition))
        sim_data[sim] = np.hstack((np.expand_dims(rt, axis=1), np.expand_dims(resp, axis=1), condition))
    return sim_data
