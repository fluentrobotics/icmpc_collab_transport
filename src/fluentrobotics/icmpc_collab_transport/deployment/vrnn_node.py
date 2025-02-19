import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import rclpy.time
import torch
from cooperative_transport.custom_rewards import custom_reward_function
from cooperative_transport.planner_utils import tf2model, tf2sim
from cooperative_transport.vrnn import VRNN
from pytorch_mppi import MPPI

from fluentrobotics.icmpc_collab_transport import logger
from fluentrobotics.icmpc_collab_transport.core.se2_types import Transform, Velocity

from .diff_drive_kinematics import kinematics
from .environment_bounds import Bounds
from .mpc_node_base import MpcNodeBase


def sim2real(sim: Transform) -> Transform:
    return Transform(
        -(sim.y * Bounds.REAL_X_RANGE / Bounds.SIM_Y_RANGE - Bounds.REAL_X_MAX),
        -(sim.x * Bounds.REAL_Y_RANGE / Bounds.SIM_X_RANGE - Bounds.REAL_Y_MAX),
        sim.theta - math.pi,
    )


def real2sim(real: Transform) -> Transform:
    return Transform(
        -(real.y - Bounds.REAL_Y_MAX) * Bounds.SIM_X_RANGE / Bounds.REAL_Y_RANGE,
        -(real.x - Bounds.REAL_X_MAX) * Bounds.SIM_Y_RANGE / Bounds.REAL_X_RANGE,
        real.theta + math.pi,
    )


class VRNNWrapper:
    def __init__(self, goal: Transform, obstacles: list[Transform]) -> None:
        """
        Goal and obstacles are specified in real coordinates.
        """
        goal = real2sim(goal)
        obstacles = [real2sim(obstacle) for obstacle in obstacles]

        self.device = torch.device("cpu")
        self.goal = torch.tensor(
            [goal.x, goal.y], dtype=torch.float32, device=self.device
        )
        self.obstacles = torch.empty(
            (len(obstacles), 2), dtype=torch.float32, device=self.device
        )
        for idx, obstacle in enumerate(obstacles):
            self.obstacles[idx, 0] = obstacle.x
            self.obstacles[idx, 1] = obstacle.y

        self.model = VRNN.load_from_checkpoint(
            Path("src/eleyng/table-carrying-ai/model.ckpt"),
            map_location=self.device,
        )
        self.model.eval()

        # https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/configs/robot/robot_planner_config.py#L28
        # NOTE: model.H is not overwritten in the original code
        self.H = 30
        # https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/configs/robot/robot_planner_config.py#L33
        # NOTE: model.seq_len is not overwritten in the original code
        self.SEQ_LEN = 120
        # https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/configs/robot/robot_planner_config.py#L37
        # https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/libs/hil_real_robot.py#L252
        self.BSIZE = 16
        self.model.batch_size = self.BSIZE
        # https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/configs/robot/robot_planner_config.py#L39
        # https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/libs/hil_real_robot.py#L252
        self.SKIP = 1
        self.model.skip = self.SKIP

        self.state_history: deque[torch.Tensor] = deque(maxlen=self.H // self.SKIP + 1)
        self.cached_best_sample: torch.Tensor | None = None

    def add_state(self, state_real: Transform) -> None:
        state_sim = real2sim(state_real)

        # https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/cooperative_transport/gym_table/envs/table_env.py#L1013
        state = torch.zeros(
            9, dtype=torch.float32, device=self.device, requires_grad=False
        )
        state[0] = state_sim.x
        state[1] = state_sim.y
        state[2] = math.cos(state_sim.theta)
        state[3] = math.sin(state_sim.theta)
        state[4] = self.goal[0]
        state[5] = self.goal[1]
        dist2obs = torch.linalg.norm(self.obstacles - state[:2], dim=1)
        most_relevant_obs_idx = torch.argmin(dist2obs)
        most_relevant_obs = self.obstacles[most_relevant_obs_idx]
        state[6] = most_relevant_obs[0]
        state[7] = most_relevant_obs[1]
        state[8] = state_sim.theta

        self.state_history.append(state)
        # VRNN requires the state history to be saturated, even at the start of
        # the interaction. In the original code, the state history was
        # bootstrapped by feeding "ground truth" actions from the collected
        # dataset into simulator dynamics, but we obviously don't have access to
        # ground truth information here. We'll fill the state history with the
        # same state, i.e., the system is stationary, as this is trivially
        # valid.
        while (
            self.state_history.maxlen is not None
            and len(self.state_history) < self.state_history.maxlen
        ):
            self.state_history.append(state)

    @torch.no_grad()
    def get_nominal_rollout(self) -> tuple[torch.Tensor, list[Transform]]:
        """
        Returns a rollout in real coordinates. Tensor size: (90, 3)
        """

        # https://github.com/eleyng/table-carrying-ai/blob/ee3339363b6461bad86da6ba5f573ea4eef7dabe/libs/hil_real_robot.py#L592
        # NOTE: Control queue (u_queue) is not used if interaction forces are
        # not used. To our knowledge, interaction forces were not used in the
        # original evaluation for model inputs (usage flag is set to False
        # everywhere).
        state_history = torch.vstack(tuple(self.state_history))

        model_input = tf2model(state_history, self.obstacles).repeat(self.BSIZE, 1, 1)
        samples = self.model.sample(model_input, seq_len=self.SEQ_LEN)
        waypoints = tf2sim(samples, state_history, self.H)
        eval = np.sum(
            np.array(
                [
                    custom_reward_function(
                        waypoints[i, :, :4].detach().cpu().numpy(),
                        self.goal.detach().cpu().numpy(),
                        self.obstacles.detach().cpu().numpy(),
                        vectorized=True,
                    )
                    for i in range(waypoints.shape[0])
                ]
            ),
            axis=-1,
        )

        best_sample_idx = np.argmax(eval)
        best_sample = waypoints[best_sample_idx, :, :]

        # Convert model output from sim to real coordinates
        """
        NOTE:
        The original code's implementation of odometry in tf2sim appears to not
        preserve the Pythagorean trigonometric identity
            `sin(theta)^2 + cos(theta)^2 = 1`,
        since it simply calculates the prefix sum of delta_cos and delta_sin.
        This causes problems when reconstructing the angle of waypoint states
        using
            `theta = atan2( sin(theta), cos(theta) )`.

        A correct implementation of odometry would be the following:
        |   Assume theta_0 is known
        |
        |   for i in [1, n]:
        |       From tf2model,
        |           delta_cos_i := cos(theta_i) - cos(theta_i-1)
        |           delta_sin_i := sin(theta_i) - sin(theta_i-1)
        |       Thus,
        |           cos(theta_i) = delta_cos_i + cos(theta_i-1)
        |           sin(theta_i) = delta_sin_i + sin(theta_i-1)
        \_          theta_i = atan2( sin(theta_i), cos(theta_i) )

        However, the model's prediction of delta_cos and delta_sin is not
        constrained properly, so a correct implementation of odometry still
        produces values that do not satisfy the first property.

        Below, we simply assume that the orientation of the object does not
        change.
        """
        sample_states_real: list[torch.Tensor] = []
        as_transforms: list[Transform] = []
        for idx in range(best_sample.shape[0]):
            state_sim = Transform(
                best_sample[idx, 0].item(),
                best_sample[idx, 1].item(),
                state_history[-1, 8].item(),
                # math.atan2(best_sample[idx, 3].item(), best_sample[idx, 2].item()),
            )
            state_real = sim2real(state_sim)
            sample_states_real.append(
                torch.tensor(
                    [state_real.x, state_real.y, state_real.theta],
                    dtype=torch.float32,
                    device="cpu",
                    requires_grad=False,
                )
            )
            as_transforms.append(state_real)

        best_sample_real = torch.vstack(sample_states_real)
        return best_sample_real, as_transforms


class VrnnNode(MpcNodeBase):
    def __init__(self) -> None:
        super().__init__("vrnn_mppi")

        cov = torch.eye(2, dtype=torch.float32)
        cov[0, 0] = 0.25
        cov[1, 1] = 1

        self.mppi = MPPI(
            kinematics,
            self.cost,
            3,
            cov,
            num_samples=100,
            terminal_state_cost=self.terminal_state_cost,
            u_min=torch.tensor([-0.0, -1], dtype=torch.float32),
            u_max=torch.tensor([0.3, 1], dtype=torch.float32),
        )

        self.vrnn_wrapper = VRNNWrapper(self.goal, [self.obstacle])
        self.vrnn_prediction: torch.Tensor | None = None
        self.tf_R_obj: Transform | None = None

        self.timer = self.create_timer(1 / 15, self.timer_callback)

    def cost(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Input:
        s: robot global state  (shape: NUM_SAMPLES x 3)
        a: robot action   (shape: NUM_SAMPLES x 2)

        Output:
        cost associated with each state-action pair (shape: NUM_SAMPLES)
        """
        assert s.ndim == 2 and s.shape[-1] == 3
        assert a.ndim == 2 and a.shape[-1] == 2
        assert self.vrnn_prediction is not None

        # Evaluated in terminal_state_cost
        return torch.zeros(s.shape[0], dtype=torch.float32)

    def terminal_state_cost(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Input:
        s: robot global state  (shape: 1 x NUM_SAMPLES x HORIZON x 3)
        a: robot action   (shape: 1 x NUM_SAMPLES x HORIZON x 2)

        Output:
        cost associated with each state-action pair (shape: NUM_SAMPLES)
        """
        assert s.ndim == 4 and s.shape[0] == 1 and s.shape[-1] == 3
        assert a.ndim == 4 and a.shape[0] == 1 and a.shape[-1] == 2
        assert self.vrnn_prediction is not None
        assert self.tf_R_obj is not None

        cost = torch.zeros(s.shape[1], dtype=torch.float32)

        for state_idx in range(s.shape[2]):
            rollout_state = s[0, :, state_idx, :]
            assert rollout_state.ndim == 2 and rollout_state.shape[1] == 3
            nominal_state = self.vrnn_prediction[state_idx * 3]
            assert nominal_state.shape == (3,)

            nominal_state_robot = torch.empty_like(nominal_state)
            tf_map_obj_ = Transform(
                nominal_state[0].item(),
                nominal_state[1].item(),
                nominal_state[2].item(),
            )
            tf_map_R_ = tf_map_obj_ @ self.tf_R_obj.inverse()
            nominal_state_robot[0] = tf_map_R_.x
            nominal_state_robot[1] = tf_map_R_.y
            nominal_state_robot[2] = tf_map_R_.theta

            cost += (0.95**state_idx) * torch.linalg.norm(
                rollout_state[:, :2] - nominal_state_robot[:2], dim=1
            )

        return cost

    def timer_callback(self) -> None:
        _fn_entry_time = time.time()

        # Get the latest global pose
        tf_map_R = self.tf2_wrapper.get_latest_pose("map", "base_link")
        if tf_map_R is None:
            logger.warning("tf_map_R is not available")
            return

        tf_R_wr = self.tf2_wrapper.get_latest_pose("base_link", "link_wrist_yaw")
        if tf_R_wr is None:
            logger.warning("tf_R_wr is not available")
            return
        tf_R_wr.theta = 0

        if self.latest_wrist_theta is None:
            logger.warning("Joint state data is not available")
            return

        theta_wr_obj = self.latest_wrist_theta
        tf_wr_obj = Transform(
            self.object_length / 2 * math.sin(theta_wr_obj),
            -self.object_length / 2 * math.cos(theta_wr_obj),
            theta_wr_obj,
        )

        tf_R_obj = tf_R_wr @ tf_wr_obj
        tf_map_obj = tf_map_R @ tf_R_wr @ tf_wr_obj

        self.vrnn_wrapper.add_state(tf_map_obj)
        self.publish_common_poses(tf_R_obj)

        if self.latest_robot_vel is None:
            logger.warning("Robot odom is not available")
            return

        # Convert pose to MPPI state representation
        mppi_state = torch.tensor(
            [
                tf_map_R.x,
                tf_map_R.y,
                tf_map_R.theta,
            ],
            dtype=torch.float32,
        )

        if (self.goal.inverse() @ tf_map_R).dist < 0.1 or (
            self.goal.inverse() @ tf_map_obj
        ).dist < 0.25:
            # Termination condition
            action = torch.tensor([0, 0], dtype=torch.float32)
        else:
            # Get best action from MPPI
            self.vrnn_prediction, nominal_path = self.vrnn_wrapper.get_nominal_rollout()
            self.tf_R_obj = tf_R_obj

            self.mppi.u_init = torch.tensor(
                [self.latest_robot_vel.x, self.latest_robot_vel.omega],
                dtype=torch.float32,
            )
            action = self.mppi.command(mppi_state)

            rollouts = self.mppi.states
            costs = self.mppi.cost_total
            if rollouts is not None and costs is not None:
                self.vis_utils.visualize_rollouts(rollouts, costs)
                self.vis_utils.visualize_path(nominal_path)
                self.vis_utils.visualize_path2(
                    [tf_map_obj @ tf_R_obj.inverse() for tf_map_obj in nominal_path]
                )

        # Publish action
        self.publish_cmd_vel(
            Velocity(
                x=action[0].item(),
                omega=action[1].item(),
            )
        )

        _fn_exit_time = time.time()
        logger.debug(f"{(_fn_exit_time - _fn_entry_time) * 1000:.3f} ms")


if __name__ == "__main__":
    rclpy.init()

    node = VrnnNode()

    rclpy.spin(node)
    rclpy.shutdown()
