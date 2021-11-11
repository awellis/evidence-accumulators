import numpy as np
import matplotlib.pyplot as plt

def velocity(t, A=1, f=0.5):
	return A * 1/(2 * np.pi * f) * (1 - np.cos(2 * np.pi * f * t))

def acceleration(t, A=1, f=0.5):
	return A * np.sin(2 * np.pi * f * t)	


t = np.linspace(0, 2, 100)

plt.plot(t, velocity(t))
plt.plot(t, acceleration(t))
plt.show()