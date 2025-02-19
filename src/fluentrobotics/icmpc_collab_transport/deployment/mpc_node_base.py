import torch
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState

from fluentrobotics.icmpc_collab_transport.core.se2_types import Transform, Velocity

from .tf2_wrapper import TF2Wrapper
from .vis_utils import VisualizationUtils


class MpcNodeBase(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        self.goal = Transform(0.5, -2.5)
        self.obstacle = Transform(0.5, 0.0)
        self.object_length = 3.0 / 3.28084  # meters

        self.goal_tensor = torch.tensor([self.goal.x, self.goal.y], dtype=torch.float32)
        self.obstacle_tensor = torch.tensor(
            [self.obstacle.x, self.obstacle.y], dtype=torch.float32
        )

        self.tf2_wrapper = TF2Wrapper(self)
        self.vis_utils = VisualizationUtils(self)

        self.latest_wrist_theta: float | None = None
        self.latest_wrist_omega: float | None = None
        self.joint_state_sub = self.create_subscription(
            JointState, "/stretch/joint_states", self.joint_state_callback, 1
        )

        self.latest_robot_vel: Velocity | None = None
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 1
        )

        self.cmd_vel_pub = self.create_publisher(TwistStamped, "/stretch/cmd_vel", 1)
        self.human_vel_pub = self.create_publisher(TwistStamped, "/human_1/twist", 1)

    def joint_state_callback(self, msg: JointState) -> None:
        joint_wrist_yaw_idx = msg.name.index("joint_wrist_yaw")
        self.latest_wrist_theta = msg.position[joint_wrist_yaw_idx]
        self.latest_wrist_omega = msg.velocity[joint_wrist_yaw_idx]

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_robot_vel = Velocity(
            msg.twist.twist.linear.x, 0, msg.twist.twist.angular.z
        )

    def publish_cmd_vel(self, vel: Velocity) -> None:
        command = TwistStamped()
        command.header.frame_id = "base_link"
        command.header.stamp = self.get_clock().now().to_msg()
        command.twist.linear.x = vel.x
        command.twist.angular.z = vel.omega

        self.cmd_vel_pub.publish(command)

        # Publish observed human action (for vis and analysis)
        if self.tf2_wrapper.latest_human_velocity is not None:
            command.header.frame_id = "human_1"
            command.twist.linear.x = self.tf2_wrapper.latest_human_velocity.x
            command.twist.linear.y = self.tf2_wrapper.latest_human_velocity.y
            command.twist.angular.z = self.tf2_wrapper.latest_human_velocity.omega
            self.human_vel_pub.publish(command)

    def publish_common_poses(self, tf_R_obj: Transform) -> None:
        self.tf2_wrapper.publish_2d_poses(
            [
                ("map", "goal", self.goal),
                ("map", "obstacle", self.obstacle),
                ("base_link", "object_center", tf_R_obj),
            ]
        )
