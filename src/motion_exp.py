import numpy as np
from motion_simulation import *

def motion_experiment(n_trials, n_unique_motions, motion_dur):
    """
    Quick and dirty simulation of a motion experiment with
    n_unique_motions different motion intensities randomly mixed among n_trials.
    """
    if n_trials % (n_unique_motions*2) != 0:
        raise ValueError("n_trials has to be a multiple of n_trials_per_motion * 2!")

    n_trials_per_motion = n_trials / (n_unique_motions*2)

    motion_set = np.empty((n_unique_motions, int(motion_dur*1000)))

    # sample intensities for uniform distribution
    motion_intensities = np.random.uniform(0.5, 5, size=n_unique_motions)

    # get acceleration for each motion intensity
    for i in range(n_unique_motions):
        motion_set[i] = acceleration(motion_dur, motion_intensities[i], motion_dur/motion_dur**2) **2

    # add same motion profiles but in other direction to the motion set
    motion_set = np.append(motion_set, -1 * motion_set, axis=0)
    # duplicate motion set to get n_trials
    out = np.repeat(motion_set, n_trials_per_motion, axis=0)
    # shuffle trials
    np.random.shuffle(out)

    return out