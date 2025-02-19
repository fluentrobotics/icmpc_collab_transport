# MIT License
#
# Copyright (c) 2023 Eley Ng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file was originally published here:
# https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/cooperative_transport/gym_table/envs/custom_rewards.py
#
# Modifications:
#   - Removed unused functions.
#   - Removed comment block and interaction forces logic.
#       - NOTE: Interaction forces are unused in the original code:
#         https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/cooperative_transport/gym_table/envs/custom_rewards.py

import numpy as np

from cooperative_transport.constants import WINDOW_H, WINDOW_W

## Define custom reward functions here
def custom_reward_function(states, goal, obs, env=None, vectorized=False, interaction_forces=False, skip=5, u_r=None, u_h=None, collision=None, collision_checking_env=None, success=None):
    # states should be an N x state_dim array
    assert (
        len(states.shape) == 2
    ), "state shape mismatch for compute_reward. Expected (n, {0}), where n is the set of states you are evaluating. Got {1}".format(
        states.shape
    )

    assert states is not None, "states parameter cannot be None"

    n = states.shape[0]
    reward = np.zeros(n)
    # slack reward
    reward += -0.1

    dg = np.linalg.norm(states[:, :2] - goal, axis=1)

    sigma_g = 300
    r_g = np.exp(-np.power(dg, 2) / (2 * sigma_g ** 2))
    reward += r_g

    r_obs = np.zeros(n)
    sigma_o = 50

    num_obstacles = obs.shape[0]
    if states is not None:
        d2obs_lst = np.asarray(
            [
                np.linalg.norm(states[:, :2] - obs[i, :], axis=1)
                for i in range(num_obstacles)
            ],
            dtype=np.float32,
        )

    # negative rewards for getting close to wall
    for i in range(num_obstacles):
        d = d2obs_lst[i]
        r_obs += - np.exp(-np.power(d, 2) / (2 * sigma_o ** 2))

    r_obs += - np.exp(-np.power((states[:, 0] - 0), 2) / (2 * sigma_o ** 2))
    r_obs += - np.exp(-np.power((states[:, 0] - WINDOW_W), 2) / (2 * sigma_o ** 2))
    r_obs += - np.exp(-np.power((states[:, 1] - 0), 2) / (2 * sigma_o ** 2))
    r_obs += - np.exp(-np.power((states[:, 1] - WINDOW_H), 2) / (2 * sigma_o ** 2))

    reward += r_obs

    if not vectorized:
        return reward[0]
    else:
        return reward
