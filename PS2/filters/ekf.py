"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma
        V_t =np.array([[(-u[1] * np.sin(u[0] + self._state_bar.mu[2]))[0], (np.cos(u[0] + self._state_bar.mu[2]))[0], 0],
                  [(u[1] * np.cos(u[0] + self._state_bar.mu[2]))[0], (np.sin(u[0] + self._state_bar.mu[2]))[0], 0],
                  [1, 0, 1]])

        G_t = np.array([[1, 0, (-u[1] * np.sin(u[0] + self._state_bar.mu[2]))[0]],
                         [0, 1, (u[1] * np.cos(u[0] + self._state_bar.mu[2]))[0]],
                         [0, 0, 1]])
        R_t = V_t @ get_motion_noise_covariance(u, self._alphas) @ V_t.T

        self._state_bar.mu = get_prediction(self._state_bar.mu[:, 0], u)[np.newaxis].T
        self._state_bar.mu[2, 0] = wrap_angle(self._state_bar.mu[2, 0])
        self._state_bar.Sigma = G_t @ self._state_bar.Sigma @ G_t.T + R_t
    def update(self, z):
        # TODO implement correction step
        H_t = np.array([(self._field_map.landmarks_poses_y[int(z[1])] - self._state_bar.mu[1, 0]) / ((self._field_map.landmarks_poses_x[int(z[1])] - self._state_bar.mu[0, 0]) ** 2 + (self._field_map.landmarks_poses_y[int(z[1])] - self._state_bar.mu[1, 0]) ** 2),
                       -(self._field_map.landmarks_poses_x[int(z[1])] - self._state_bar.mu[0, 0]) / ((self._field_map.landmarks_poses_x[int(z[1])] - self._state_bar.mu[0, 0]) ** 2 + (self._field_map.landmarks_poses_y[int(z[1])] - self._state_bar.mu[1, 0]) ** 2),
                       -1])

        S_t = H_t @ self._state_bar.Sigma @ H_t.T + self._Q * 2
        K_t = self._state_bar.Sigma @ H_t.T * S_t ** (-1)

        self._state.mu = self._state_bar.mu + (K_t * wrap_angle(z[0] - get_expected_observation(self.mu_bar, int(z[1]))[0]))[np.newaxis].T
        self._state.mu[2, 0] = wrap_angle(self._state.mu[2, 0])
        self._state.Sigma = np.asarray((np.eye(3) - np.asmatrix(K_t).T @ np.asmatrix(H_t)) @ self._state_bar.Sigma)









