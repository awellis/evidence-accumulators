import numpy as np
import math

def acceleration(duration, A, f, dt=0.001):
    t = np.arange(0, duration, dt)
    motion_profile = A * np.sin(2 * math.pi * f * t)
    return motion_profile

def velocity(duration, A, f, dt=0.001):
    t = np.arange(0, duration, dt)
    motion_profile = A * 1/(2*math.pi*f) * (1 - np.cos(2 * math.pi * f * t))
    return motion_profile