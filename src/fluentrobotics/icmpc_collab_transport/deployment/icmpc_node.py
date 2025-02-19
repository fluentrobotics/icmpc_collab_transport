import math
import time

import rclpy
import rclpy.time
import torch
from pytorch_mppi import MPPI
from std_msgs.msg import String

from fluentrobotics.icmpc_collab_transport import logger
from fluentrobotics.icmpc_collab_transport.core.passing_inference import (
    ActionLikelihoodDistribution,
    InferenceWrapper,
    PosteriorDistribution,
    PriorDistribution,
)
from fluentrobotics.icmpc_collab_transport.core.se2_types import Transform, Velocity

from .diff_drive_kinematics import kinematics
from .mpc_node_base import MpcNodeBase


class IcmpcNode(MpcNodeBase):
    def __init__(self) -> None:
        super().__init__("icmpc")

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

        self.inference_wrapper = InferenceWrapper(self.obstacle, self.goal)

        # Global state needed by cost function
        self.tf_map_R: Transform | None = None
        self.tf_R_obj: Transform | None = None
        self.v_map_H: Velocity | None = None

        self.timer = self.create_timer(1 / 15, self.timer_callback)

        self.pmf_pub = self.create_publisher(String, "/passing_strategy/pmf", 1)

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

        cost = torch.zeros(s.shape[0], dtype=torch.float32)

        clearance = 0.5
        cost += torch.relu(
            -torch.log2(
                torch.linalg.vector_norm(
                    s[:, 0:2] - self.obstacle_tensor, keepdim=False, dim=1
                )
                / clearance
            )
        )

        assert self.tf_map_R is not None
        assert self.tf_R_obj is not None
        assert self.v_map_H is not None

        for sample_idx in range(s.shape[0]):
            tf_map_R = Transform(
                s[sample_idx, 0].item(),
                s[sample_idx, 1].item(),
                s[sample_idx, 2].item(),
            )

            tf_map_object = tf_map_R @ self.tf_R_obj

            action = tf_map_R @ Velocity(x=1.0)

            prior = PriorDistribution(self.obstacle, self.goal, tf_map_object)
            action_likelihood = ActionLikelihoodDistribution(
                self.obstacle, self.goal, tf_map_object
            )
            H = PosteriorDistribution(
                prior, action_likelihood, self.v_map_H, action
            ).get_entropy()
            cost[sample_idx] += H

        return cost

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

        return (
            torch.linalg.vector_norm(
                s[0, :, -1, 0:2] - self.goal_tensor, keepdim=False, dim=1
            )
            ** 2
        )

    def timer_callback(self) -> None:
        _fn_entry_time = time.time()

        self.tf_map_R = self.tf2_wrapper.get_latest_pose("map", "base_link")
        if self.tf_map_R is None:
            logger.warning("tf_map_R is not available")
            return

        tf_R_wr = self.tf2_wrapper.get_latest_pose("base_link", "link_wrist_yaw")
        if tf_R_wr is None:
            logger.warning("tf_R_wr is not available")
            return
        tf_R_wr.theta = 0

        tf_map_H = self.tf2_wrapper.get_latest_pose("map", "human_1")
        if tf_map_H is None:
            logger.warning("tf_map_H is not available")
            return

        if self.latest_wrist_theta is None:
            logger.warning("Joint state data is not available")
            return

        theta_wr_obj = self.latest_wrist_theta

        tf_wr_obj = Transform(
            self.object_length / 2 * math.sin(theta_wr_obj),
            -self.object_length / 2 * math.cos(theta_wr_obj),
            theta_wr_obj,
        )

        tf_wr_Hgrasp = Transform(
            self.object_length * math.sin(theta_wr_obj),
            -self.object_length * math.cos(theta_wr_obj),
            theta_wr_obj,
        )

        tf_map_Hgrasp = self.tf_map_R @ tf_R_wr @ tf_wr_Hgrasp
        tf_map_obj = self.tf_map_R @ tf_R_wr @ tf_wr_obj

        self.tf_R_obj = tf_R_wr @ tf_wr_obj
        self.publish_common_poses(self.tf_R_obj)
        #############################################

        # Convert pose to MPPI state representation
        mppi_state = torch.tensor(
            [
                self.tf_map_R.x,
                self.tf_map_R.y,
                self.tf_map_R.theta,
            ],
            dtype=torch.float32,
        )

        #############################################
        # Calculate Velocities
        if self.latest_robot_vel is None:
            logger.warning("Robot odom is not available")
            return
        v_R = self.latest_robot_vel
        v_map_R = self.tf_map_R @ v_R

        v_H = self.tf2_wrapper.latest_human_velocity
        if v_H is None:
            logger.warning("Human mocap velocity is not available")
            return

        if math.sqrt(v_H.x**2 + v_H.y**2) < 0.1:
            logger.warning("LOW VELOCITY")

        self.v_map_H = tf_map_H @ v_H

        self.inference_wrapper.add_object_state(tf_map_obj)
        self.inference_wrapper.update_h_action(self.v_map_H)
        self.inference_wrapper.update_r_action(v_map_R)

        pmf_msg = "\n".join(
            [
                self.inference_wrapper.get_prior_str(),
                self.inference_wrapper.get_action_likelihood_str(),
                self.inference_wrapper.get_posterior_str(),
            ]
        )
        self.pmf_pub.publish(String(data=pmf_msg))

        if (self.goal.inverse() @ self.tf_map_R).dist < 0.1 or (
            self.goal.inverse() @ tf_map_obj
        ).dist < 0.25:
            # Termination condition
            action = torch.tensor([0, 0], dtype=torch.float32)
        else:
            # Get best action from MPPI
            self.mppi.u_init = torch.tensor(
                [v_R.x, v_R.omega],
                dtype=torch.float32,
            )
            action = self.mppi.command(mppi_state)

            rollouts = self.mppi.states
            costs = self.mppi.cost_total
            if rollouts is not None and costs is not None:
                self.vis_utils.visualize_rollouts(rollouts, costs)

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

    node = IcmpcNode()

    rclpy.spin(node)
    rclpy.shutdown()
