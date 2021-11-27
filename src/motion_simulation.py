import numpy as np
from motion_simulation import *
from tensorflow.python.keras.utils.np_utils import to_categorical

def acceleration(duration, A, f, dt=0.001):
    t = np.arange(0, duration, dt)
    motion_profile = A * np.sin(2 * np.pi * f * t)
    return motion_profile

def velocity(duration, A, f, dt=0.001):
    t = np.arange(0, duration, dt)
    motion_profile = A * 1/(2*math.pi*f) * (1 - np.cos(2 * np.pi * f * t))
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
    motion_intensities = np.random.uniform(0.5, 5, size=n_unique_motions)

    # get acceleration for each motion intensity
    for i in range(n_unique_motions):
        motion_set[i] = acceleration(motion_dur, motion_intensities[i], motion_dur/motion_dur**2) **2

    # add same motion profiles but in other direction to the motion set
    motion_set = np.append(motion_set, -1 * motion_set, axis=0)
    # add condition index
    motion_set = np.c_[motion_set,np.arange(n_unique_motions*2)]
    # duplicate motion set to get n_trials
    motion_set = np.repeat(motion_set, n_trials_per_motion, axis=0)
    # shuffle trials
    np.random.shuffle(motion_set)

    # get array with conditions
    condition = to_categorical(motion_set[:, motion_dur*1000])

    return motion_set[:, :motion_dur*1000], condition
