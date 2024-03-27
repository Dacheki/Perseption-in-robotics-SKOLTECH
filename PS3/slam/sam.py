"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

import numpy as np
import mrob
from scipy.linalg import inv
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance
from tools.jacobian import state_jacobian, inverse_observation_jacobian
from scipy.linalg import inv


class Sam(SlamBase):
    def __init__(self, initial_state, alphas, state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, *args, **kwargs):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas


        self.graph = mrob.FGraph()
        #added part
        print('Task 1A')

        self.mu = initial_state.mu
        self.sigma = initial_state.Sigma
        self.sigma = inv(self.sigma)
        self.pose = self.graph.add_node_pose_2d(self.mu)
        self.graph.add_factor_1pose_2d(self.mu, self.pose, self.sigma)
        self.chi2 = []
        self.graph.print(True)

        self.landmarks = {}
        self.landmarks_node = []
        self.skip = []
        self.estimate = {}

    def predict(self, u):
        print('Task1B')
        print('estimated-state_1',self.graph.get_estimated_state())
        self.pose_for_jac = self.graph.get_estimated_state()[self.pose].T[0]
        J = state_jacobian(self.pose_for_jac, u)[1]

        new_node = self.graph.add_node_pose_2d(np.zeros(3))

        W_u = J @ get_motion_noise_covariance(u, self.alphas) @ J.T

        self.graph.add_factor_2poses_2d_odom(u,self.pose, new_node, inv(W_u))
        self.pose = new_node
        print('estimated-state_2', self.graph.get_estimated_state())

    def update(self, z, Q):
        print('Task1C')
        print('estimated-state_2', self.graph.get_estimated_state())
        W_z = Q
        for z_i in z:
            if z_i[-1] in self.landmarks.keys():
                initializeLandmark = False
                self.graph.add_factor_1pose_1landmark_2d(z_i[:2], self.pose, self.landmarks[(z_i[-1])],
                                                         inv(W_z), initializeLandmark=initializeLandmark)
                # print(self.graph.get_estimated_state())

            else:
                initializeLandmark = True
                land_mark = self.graph.add_node_landmark_2d(np.zeros(2))
                self.landmarks[z_i[2]] = land_mark
                self.graph.add_factor_1pose_1landmark_2d(z_i[:2], self.pose, self.landmarks[(z_i[-1])],
                                                         inv(W_z), initializeLandmark=initializeLandmark)
                self.landmarks_node.append(
                    f'landmark: {self.landmarks[z_i[-1]]},estimate state: {self.graph.get_estimated_state()[self.landmarks[(z_i[-1])]].T[0]}')
                self.skip.append(self.landmarks[z_i[-1]])
                # print(
                #     f'Node: {self.landmarks.get(z_i[-1])}, Pos:{self.graph.get_estimated_state()[self.landmarks.get(z_i[-1])].T}')


                # print(self.landmarks.get(z_i[-1]), self.graph.get_estimated_state())

        for i in range(len(self.graph.get_estimated_state())):
            if i not in (self.skip):
                print(f'Node: {i},Pos: {self.graph.get_estimated_state()[i]}')

        print("\n".join(self.landmarks_node))
        #
        print('inform martix', inv(W_z))





    def solve(self, method):
          print('TaskD')
          self.graph.solve(method)
          for i in range(len(self.graph.get_estimated_state())):
              print(f'Node: {i},Pos: {self.graph.get_estimated_state()[i]}')








