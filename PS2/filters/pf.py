"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, bearing_std, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, bearing_std)

        self.num_particles = num_particles
        self.weight = np.zeros(self.num_particles)
        self.particles = np.random.multivariate_normal(self.mu, self.Sigma, num_particles)
    def predict(self, u):
        for i in range(self.num_particles):
            self.particles[i, :] = sample_from_odometry(self.particles[i, :], u, self._alphas)
            self.particles[i, 2] = wrap_angle(self.particles[i, 2])

        self._state_bar = get_gaussian_statistics(self.particles)
    def update(self, z):
        z_bar = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            z_bar[i] = get_observation(self.particles[i], int(z[1]))[0]

        for i in range(self.num_particles):
            self.weight[i] = gaussian.pdf(z_bar[i], loc=z[0], scale=np.sqrt(self._Q))
            if self.weight[i] == 0:
                self.weight[i] = 1.e-200
        self.weight /= sum(self.weight)

        R = uniform(0, 1 / self.num_particles)
        t = self.weight[0]
        X = np.empty((self.num_particles, self.state_dim))
        ind = 0
        for m in range(self.num_particles):
            U = R + m / self.num_particles
            while U > t:
                ind += 1
                t += self.weight[ind]
            X[m, :] = self.particles[ind, :]

        self.particles = X
        self._state = get_gaussian_statistics(self.particles)