import math

import torch
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from fluentrobotics.icmpc_collab_transport.core.se2_types import Transform


class VisualizationUtils:
    def __init__(self, node: Node) -> None:
        self._node = node

        self._rollouts_pub = self._node.create_publisher(
            MarkerArray, "/mppi/vis/rollouts", 1
        )

        self._path_pub = self._node.create_publisher(
            Path, f"/{self._node.get_name()}/vis/path_object_center", 1
        )
        self._path_pub2 = self._node.create_publisher(
            Path, f"/{self._node.get_name()}/vis/path_robot", 1
        )

    def visualize_rollouts(self, rollouts: torch.Tensor, costs: torch.Tensor) -> None:
        """
        Input:
        rollouts: (shape: 1 x NUM_SAMPLES x HORIZON x 3)
        costs: (shape: NUM_SAMPLES)
        """
        assert rollouts.ndim == 4 and rollouts.shape[0] == 1 and rollouts.shape[-1] == 3

        min_cost = torch.min(costs).item()
        max_cost = torch.max(costs).item()

        marker_array = MarkerArray()
        for sample_idx in range(rollouts.shape[1]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = sample_idx
            marker.type = Marker.LINE_STRIP
            marker.scale.x = 0.01
            marker.lifetime = Duration(seconds=1.0).to_msg()  # type: ignore

            cost = costs[sample_idx].item()
            if cost == min_cost:
                marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
            else:
                cost_prop = (cost - min_cost) / (max_cost - min_cost)
                # Smooth transition from red -> yellow -> green
                # prop: 1.0 -> 0.5 -> 0.0
                # r   : 1.0 -> 1.0 -> 0.0
                # g   : 0.0 -> 1.0 -> 1.0
                marker.color.r = min(1.0, 2 * cost_prop)
                marker.color.g = max(0.0, 1 - 2 * cost_prop)
                marker.color.a = 0.5

            for state_idx in range(rollouts.shape[2]):
                state = rollouts[0, sample_idx, state_idx]
                marker.points.append(Point(x=state[0].item(), y=state[1].item()))

            marker_array.markers.append(marker)

        self._rollouts_pub.publish(marker_array)

    def visualize_path(self, path: list[Transform]) -> None:
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self._node.get_clock().now().to_msg()

        for state in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self._node.get_clock().now().to_msg()

            pose.pose.position.x = state.x
            pose.pose.position.y = state.y
            pose.pose.orientation.w = math.cos(state.theta / 2)
            pose.pose.orientation.z = math.sin(state.theta / 2)

            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)

    def visualize_path2(self, path: list[Transform]) -> None:
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self._node.get_clock().now().to_msg()

        for state in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self._node.get_clock().now().to_msg()

            pose.pose.position.x = state.x
            pose.pose.position.y = state.y
            pose.pose.orientation.w = math.cos(state.theta / 2)
            pose.pose.orientation.z = math.sin(state.theta / 2)

            path_msg.poses.append(pose)

        self._path_pub2.publish(path_msg)
