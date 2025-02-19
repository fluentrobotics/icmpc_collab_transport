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
# https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/libs/planner/planner_utils.py
#
# Modifications:
#   - Removed unused functions.
#   - Reimplemented tf2model (see docstring).
#   - Modified tf2sim to replace numpy functions with torch equivalents.

import torch


@torch.no_grad()
def tf2model(state_history: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
    """
    This is a reimplementation of the algorithm in Sections (III, III-A) of Ng
    et al., 2022 to process a history of environment states to an 8-dimensional
    model input.

    The original implementation linked below did not follow the algorithm as
    published and encountered runtime errors. In particular, it did not
    calculate the distance to the *closest* obstacle.
    https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/libs/planner/planner_utils.py#L124
    """
    assert state_history.shape == (31, 9)
    assert obstacles.ndim == 2 and obstacles.shape[1] == 2

    device = state_history.device

    state_history_xy = state_history[:, :2]
    assert state_history_xy.shape == (31, 2)
    state_history_th = state_history[:, 2:4]
    assert state_history_th.shape == (31, 2)

    state_history_linear_odom = torch.diff(state_history_xy, dim=0)
    assert state_history_linear_odom.shape == (30, 2)
    state_history_angular_odom = torch.diff(state_history_th, dim=0)
    assert state_history_linear_odom.shape == (30, 2)
    state_history_p_ego_goal = torch.empty(
        (state_history_linear_odom.shape[0], 2), dtype=torch.float32, device=device
    )
    state_history_p_ego_closest_obstacle = torch.empty(
        (state_history_linear_odom.shape[0], 2), dtype=torch.float32, device=device
    )

    # NOTE(reimplementation): These parameters are in the original code release.
    # https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/libs/planner/planner_utils.py#L147
    qo = 8
    qg = 8

    for t in range(state_history_linear_odom.shape[0]):
        # NOTE: It probably makes more sense for the following indexing to use
        # [t+1] because of the call to torch.diff, but the original code used
        # [t]. If the delta_t is small enough, this difference shouldn't matter
        # much.
        _cos = state_history_th[t, 0]
        _sin = state_history_th[t, 1]
        _x = state_history_xy[t, 0]
        _y = state_history_xy[t, 1]
        T_map_ego = torch.tensor(
            [
                [_cos, -_sin, _x],
                [_sin, _cos, _y],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        T_ego_map = torch.linalg.inv(T_map_ego)

        p_map_obstacles = obstacles.T
        assert p_map_obstacles.ndim == 2 and p_map_obstacles.shape[0] == 2
        p_ego_obstacles = (T_ego_map[:2, :2] @ p_map_obstacles) + T_ego_map[:2, 2:]
        assert p_ego_obstacles.ndim == 2 and p_ego_obstacles.shape[0] == 2
        closest_obstacle_idx = torch.argmin(torch.linalg.norm(p_ego_obstacles, dim=0))
        state_history_p_ego_closest_obstacle[t] = (
            p_ego_obstacles[:, closest_obstacle_idx] / qo
        )

        p_map_goal = state_history[t, 4:6][:, torch.newaxis]
        assert p_map_goal.shape == (2, 1)
        p_ego_goal = (T_ego_map[:2, :2] @ p_map_goal) + T_ego_map[:2, 2:]
        assert p_ego_goal.shape == (2, 1)
        state_history_p_ego_goal[t] = p_ego_goal.flatten() / qg

    state = torch.cat(
        (
            state_history_linear_odom,
            state_history_angular_odom,
            state_history_p_ego_goal,
            state_history_p_ego_closest_obstacle,
        ),
        dim=1,
    )
    assert state.shape == (30, 8)
    return state


@torch.no_grad
def tf2sim(sample: torch.Tensor, init_state: torch.Tensor, H: int) -> torch.Tensor:
    x = init_state[-1, 0] + torch.cumsum(sample[:, H:, 0], dim=1)

    y = init_state[-1, 1] + torch.cumsum(sample[:, H:, 1], dim=1)
    cth = (
        init_state[-1, 2] + torch.cumsum(sample[:, H:, 2], dim=1)
    )
    sth = (
        init_state[-1, 3] + torch.cumsum(sample[:, H:, 3], dim=1)
    )

    x = torch.unsqueeze(x, dim=-1)
    y = torch.unsqueeze(y, dim=-1)
    cth = torch.unsqueeze(cth, dim=-1)
    sth = torch.unsqueeze(sth, dim=-1)
    waypoints_wf = torch.cat((x, y, cth, sth), dim=-1)

    return waypoints_wf
