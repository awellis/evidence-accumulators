import numpy as np
import pandas as pd
from numba import njit, prange
from motion_simulation import *
from tensorflow.python.keras.utils.np_utils import to_categorical

@njit
def acceleration(duration, A, f, dt=0.001):
    t = np.arange(0, duration, dt)
    motion_profile = A * np.sin(2 * np.pi * f * t)
    return motion_profile

@njit
def velocity(duration, A, f, dt=0.001):
    t = np.arange(0, duration, dt)
    motion_profile = A * 1/(2*np.pi*f) * (1 - np.cos(2 * np.pi * f * t))
    return motion_profile


def motion_experiment(n_trials, n_unique_motions, motion_dur):
    """
    Quick and dirty simulation of a motion experiment with
    n_unique_motions different motion intensities randomly mixed among n_trials.
    """
    if n_trials % (n_unique_motions*2) != 0:
        raise ValueError("n_trials has to be a multiple of n_trials_per_motion * 2!")

    n_trials_per_motion = n_trials / (n_unique_motions*2)

    motion_set = np.empty((n_unique_motions, int(motion_dur*1000)))

    # sample intensities from uniform distribution
    rng = np.random.default_rng(2021)
    motion_intensities = rng.uniform(0.5, 5, size=n_unique_motions)

    # get acceleration for each motion intensity
    for i in range(n_unique_motions):
        motion_set[i] = acceleration(motion_dur, motion_intensities[i], motion_dur/motion_dur**2) **2

    # add same motion profiles but in other direction to the motion set
    motion_set = np.append(motion_set, -1 * motion_set, axis=0)
    # add condition index
    motion_set = np.c_[motion_set, np.arange(n_unique_motions*2)]
    # duplicate motion set to get n_trials
    motion_set = np.repeat(motion_set, n_trials_per_motion, axis=0)
    # shuffle trials
    np.random.shuffle(motion_set)

    # get array with conditions
    condition = to_categorical(motion_set[:, motion_dur*1000])

    return motion_set[:, :motion_dur*1000], condition


@njit
def motion_experiment_manual(motion_dur, amplitude, frequency):
    """
    Pass an array with amplitudes to exactly replicate the experiment
    """
    n_obs = len(amplitude)

    motion_set = np.empty((n_obs, int(motion_dur*10000)), dtype=np.float32)
    for i in range(n_obs):
        if amplitude[i] > 0:
            motion_set[i] = np.append(acceleration(motion_dur, amplitude[i], frequency)**2, [0]*9000)
        else:
            motion_set[i] = np.append(-1 * acceleration(motion_dur, amplitude[i], frequency)**2, [0]*9000)

    condition = get_hot_encoded_amplitude(amplitude)

    return motion_set, condition


@njit
def get_hot_encoded_amplitude(amplitude):
    unique_motions = np.array([-0.725, -0.675, -0.625, -0.575, -0.525,
                               0.525,  0.575,  0.625,  0.675,  0.725], dtype=np.float32)
    condition = np.zeros((100, 10), dtype=np.int32)
    # data_subset = data[(data.instruction == 1) & (data.condition == 1)]

    for i in range(len(amplitude)):
        index = np.where(np.array(np.round(amplitude[i], 3), dtype=np.float32) == unique_motions)[0][0]
        condition[i, index] = 1

    return condition