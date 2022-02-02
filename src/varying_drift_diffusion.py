import numpy as np
from numba import njit, prange
from tensorflow.python.keras.utils.np_utils import to_categorical
from accumulators import *
from motion_simulation import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


@njit
def var_dm_simulator(theta, n_obs, motion_set, s=0.1, dt=0.001, max_iter=1e4):
    # parameters
    a = theta[0]  # boundary separation
    ndt = theta[1]  # non-decision time
    bias = theta[2]  # a priori bias
    kappa = theta[3]  # sensitivity to movement profile

    # initialize data
    resp = np.empty(n_obs, dtype=np.int32)
    rt = np.empty(n_obs, dtype=np.float32)

    # iterate over trials
    for n in range(n_obs):
        drift = kappa * motion_set[n]
        rt[n], resp[n] = varying_evidence_accumulation(
            drift, a, ndt, bias, s, dt, max_iter)

    return rt, resp


def var_dm_priors(n_sim=1):

    a = np.random.uniform(0.5, 3.0, size=n_sim)
    ndt = np.random.uniform(0.1, 0.5, size=n_sim)
    bias = np.random.uniform(0.2, 0.8, size=n_sim)
    kappa = np.random.uniform(0.0, 10.0, size=n_sim)

    theta = np.array([a, ndt, bias, kappa]).T

    return theta


def var_dm_batch_simulator(n_sim, n_obs):
    prior_samples = var_dm_priors(n_sim)

    # create an motion experiment with fixed design (temporarily)
    # maybe I could just give the acceleration peak intensity to BayesFlow instead of
    # one-hot-encoded condition 2darray.
    # n_unique_motions = 5
    # motion_dur = 2
    # motion_set, condition = motion_experiment(n_obs, n_unique_motions, motion_dur)

    unique_motions = np.array([-0.725, -0.675, -0.625, -0.575, -0.525,
                               0.525,  0.575,  0.625,  0.675,  0.725], dtype=np.float32)
    amplitude = np.repeat(unique_motions, 10)
    motion_set, condition = motion_experiment_manual(1, amplitude, 1)

    # simulate
    sim_data = var_dm_batch_simulator_wrap(
        prior_samples, motion_set, condition, n_sim, n_obs)

    # data prep
    one_hot_encoded_resp = to_categorical(sim_data[:, :, 1])
    return prior_samples, np.c_[np.expand_dims(sim_data[:, :, 0], axis=2), one_hot_encoded_resp, sim_data[:, :, 3:]]

# wrapper function for faster simulation


@njit
def var_dm_batch_simulator_wrap(prior_samples, motion_set, condition, n_sim, n_obs):
    sim_data = np.zeros(
        (n_sim, n_obs, condition.shape[1] + 2), dtype=np.float32)
    # iterate over simulations
    for sim in range(n_sim):
        # shuffle motion profile together with condition
        shuffler = np.random.permutation(len(motion_set))
        motion_set = motion_set[shuffler, :]
        condition = condition[shuffler, :]

        # simulate trials
        rt, resp = var_dm_simulator(prior_samples[sim], n_obs, motion_set)
        # sim_data[sim] = np.c_((rt, resp, condition))
        sim_data[sim] = np.hstack((np.expand_dims(
            rt, axis=1), np.expand_dims(resp, axis=1), condition[0:n_obs, :]))
    return sim_data


@njit(parallel=True)
def var_dm_simulator_pp_check(posterior_samples, amplitude, n_sim, n_obs):
    sim_data = np.zeros((n_sim, n_obs, 3), dtype=np.float32)
    motion_set, condition = motion_experiment_manual(1, amplitude, 1)

    # iterate over simulations
    for sim in prange(n_sim):
        # simulate trials
        rt, resp = var_dm_simulator(posterior_samples[sim], n_obs, motion_set)
        sim_data[sim] = np.hstack((np.expand_dims(rt, axis=1), np.expand_dims(resp, axis=1),
                                   np.expand_dims(amplitude, axis=1)))

    return sim_data

def var_dm_pp_check(emp_data, posterior_samples):
    n_sim = posterior_samples.shape[0]
    n_obs = emp_data.shape[0]
    rt_quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    amplitude = np.round(emp_data[:, 0], 3)
    unique_amplitude = np.round(np.sort(np.unique(amplitude)), 3)
    pred_resp_prop = np.empty((n_sim, 10))
    pred_rt_quantiles = np.empty((n_sim, 10, len(rt_quantiles)))
    emp_resp_prop = np.empty(10)

    # simulate data with samples from the joint posterior distribution
    pred_data = var_dm_simulator_pp_check(
        posterior_samples, amplitude, n_sim, n_obs)

    # predicted resp probablilities
    for sim in range(n_sim):
        # iterate over amplitudes
        for i in range(len(unique_amplitude)):
            tmp_data = pred_data[sim,
                                 (pred_data[sim, :, 2] == unique_amplitude[i]), :]
            pred_resp_prop[sim, i] = tmp_data[:, 1].mean()
            pred_rt_quantiles[sim, i] = np.quantile(
                tmp_data[:, 0], rt_quantiles)

    pred_resp_prop_quantiles = np.quantile(
        pred_resp_prop, [0.025, 0.5, 0.975], axis=0)
    # pred_rt_quantiles =

    # empirical resp probablilities
    for i in range(len(unique_amplitude)):
        tmp_data = emp_data[(amplitude == unique_amplitude[i]), 1]
        emp_resp_prop[i] = tmp_data.mean()

    # plotting
    plt.plot(range(len(unique_amplitude)),
             pred_resp_prop_quantiles[1], label="Predicted Mean", linestyle='dashed')
    plt.plot(range(len(unique_amplitude)), 1 - emp_resp_prop,
             color="black", label="Empirical Mean", linestyle='solid')
    plt.xticks(range(len(unique_amplitude)), unique_amplitude, rotation=45)
    plt.fill_between(range(len(unique_amplitude)), pred_resp_prop_quantiles[0], pred_resp_prop_quantiles[2],
                     alpha=0.2, label="Predictive Uncertainty")
    plt.ylim([0.0, 1.0])
    sns.despine()
    plt.legend()

    return pred_data
