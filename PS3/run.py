#!/usr/bin/python

"""
Sudhanva Sreesha
ssreesha@umich.edu
22-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
28-February-2021
"""

import contextlib
import os
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy
from tools.objects import Gaussian
from tools.plot import get_plots_figure
from tools.plot import plot_robot
from field_map import FieldMap
#from slam import SimulationSlamBase
from tools.data import generate_data as generate_input_data
from tools.data import load_data
from tools.plot import plot_field
from tools.plot import plot_observations
from tools.task import get_dummy_context_mgr
from tools.task import get_movie_writer
from tools.helpers import get_cli_args, validate_cli_args
from slam.sam import Sam
import mrob
from tools.plot import plot2dcov
from numpy.linalg import norm
from scipy.linalg import inv

def main():
    args = get_cli_args()
    validate_cli_args(args)
    alphas = np.array(args.alphas) ** 2
    beta = np.array(args.beta)
    beta[1] = np.deg2rad(beta[1])
    Q  = np.diag(beta**2)
    chi2_array = []
    mean_prior = np.array([180., 50., 0.])
    Sigma_prior = 1e-12 * np.eye(3, 3)
    initial_state = Gaussian(mean_prior, Sigma_prior)

    if args.input_data_file:
        data = load_data(args.input_data_file)
    elif args.num_steps:
        # Generate data, assuming `--num-steps` was present in the CL args.
        data = generate_input_data(initial_state.mu.T,
                                   args.num_steps,
                                   args.num_landmarks_per_side,
                                   args.max_obs_per_time_step,
                                   alphas,
                                   beta,
                                   args.dt)
    else:
        raise RuntimeError('')

    should_show_plots = True if args.animate else False
    should_write_movie = True if args.movie_file else False
    should_update_plots = True if should_show_plots or should_write_movie else False

    field_map = FieldMap(args.num_landmarks_per_side)

    fig = get_plots_figure(should_show_plots, should_write_movie)
    movie_writer = get_movie_writer(should_write_movie, 'Simulation SLAM', args.movie_fps, args.plot_pause_len)

    sam = Sam(initial_state=initial_state,alphas=alphas,slam_type=args.filter_name,data_association=args.data_association,update_type=args.update_type,Q = Q)

    with movie_writer.saving(fig, args.movie_file, data.num_steps) if should_write_movie else get_dummy_context_mgr():
        for t in tqdm(range(data.num_steps)):
            # Used as means to include the t-th time-step while plotting.
            tp1 = t + 1

            # Control at the current step.
            u = data.filter.motion_commands[t]
            # Observation at the current step.
            z = data.filter.observations[t]


            # TODO SLAM predict(u)
            sam.predict(u)
            # # #
            # # # # TODO SLAM update
            sam.update(z, Q)

            sam.solve(mrob.GN)
            #
            #manual
            sam.update(z, Q)

            state_1 = []
            state_2 = []
            st = sam.graph.get_estimated_state()

            for s in st:
                flat = list(s.flatten())
                state_1 = state_1 + flat
            state_1 = np.array(state_1)

            sam.solve(mrob.GN)

            st = sam.graph.get_estimated_state()

            for s in st:
                flat = list(s.flatten())
                state_2 = state_2 + flat
            state_2 = np.array(state_2)

            A = sam.graph.get_adjacency_matrix()
            W = sam.graph.get_W_matrix()
            I = sam.graph.get_information_matrix()

            inf_matrix = A.todense().T @ W.todense() @ A.todense()
            norm_inf_m = norm(I - inf_matrix)

            b = sam.graph.get_vector_b()

            d_x = inv(inf_matrix) @ b
            norm_d_x = norm(d_x - (state_1 - state_2))

            print('manual information matrix',norm_inf_m )
            print('diff between manual and auto', norm_d_x)
            if not should_update_plots:
                continue

            plt.cla()
            plot_field(field_map, z)
            plot_robot(data.debug.real_robot_path[t])
            plot_observations(data.debug.real_robot_path[t],
                              data.debug.noise_free_observations[t],
                              data.filter.observations[t])

            plt.plot(data.debug.real_robot_path[1:tp1, 0], data.debug.real_robot_path[1:tp1, 1], 'm')
            plt.plot(data.debug.noise_free_robot_path[1:tp1, 0], data.debug.noise_free_robot_path[1:tp1, 1], 'g')

            plt.plot([data.debug.real_robot_path[t, 0]], [data.debug.real_robot_path[t, 1]], '*r')
            plt.plot([data.debug.noise_free_robot_path[t, 0]], [data.debug.noise_free_robot_path[t, 1]], '*g')

            chi2_array.append(sam.graph.chi2())

            # TODO plot SLAM solution

            pos_rob = []
            pos_mark = []
            new_pose = sam.graph.get_estimated_state()
            for im in sam.landmarks.values():
                pos_rob.append(sam.graph.get_estimated_state()[im].T[0])
                new_pose[im] = '0'
            pos_rob = np.array(pos_rob)

            for i in range(len(sam.graph.get_estimated_state())):
                if new_pose[i] != '0':
                   pos_mark.append(new_pose[i])
            pos_mark = np.array(pos_mark)
            #
            # plt.scatter(pos_rob[:t, 0], pos_rob[:t, 1], c='b', label='Landmarks estimations')
            #
            # plt.plot(pos_mark[:t, 0], pos_mark[:t, 1], c = 'r', label='Landmarks estimations')

            if should_show_plots:
                # Draw all the plots and pause to create an animation effect.
                # plt.plot(pos_rob_x, pos_rob_y, 'r', label='Robot Estimated States')
                # plt.plot(pos_mark_x, pos_mark_y, 'bo', label='Landmarks estimations')
                plt.legend(loc='upper right')
                plt.draw()
                plt.pause(args.plot_pause_len)


    # plt.title("Adjacency matrix")
    # plt.spy(sam.graph.get_adjacency_matrix())
    # plt.show(True)
    #
    # plt.title("Information matrix")
    # plt.spy(sam.graph.get_information_matrix())
    # plt.show(True)
    #
    #iso
    # plot2dcov(sam.graph.get_estimated_state()[-1].T[0, :-1], scipy.sparse.linalg.inv(sam.graph.get_information_matrix()[-3:-1, -3:-1].todense(), 'm', 3))
    # plt.axis('equal')
    # plt.show(block=True)
    # #
    # print(sam.graph.solve(mrob.LM))

        for i in range(10):
            n = i + 1
            sam.solve(mrob.LM)
            print(f"Graph Solution: GN: i:{n}, chi2: {sam.graph.chi2()}")
        sam.solve(mrob.LM)
if __name__ == '__main__':
    main()
