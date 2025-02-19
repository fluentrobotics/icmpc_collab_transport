import math
from typing import overload

import rclpy.time
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from fluentrobotics.icmpc_collab_transport.core import se2_types


def _ros2_time_to_sec(t: rclpy.time.Time | TimeMsg) -> float:
    # why is ros2 like this :(
    if isinstance(t, TimeMsg):
        sec, nanosec = t.sec, t.nanosec
    elif isinstance(t, rclpy.time.Time):
        sec, nanosec = t.seconds_nanoseconds()
    else:
        raise ValueError()

    return float(sec) + float(nanosec) * 1e-9


class TF2Wrapper:
    def __init__(self, node: Node) -> None:
        self._node = node
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        self._tf2_pub = self._node.create_publisher(TFMessage, "/tf", 1)

        self._last_tf_map_human: TransformStamped | None = None
        self.latest_human_velocity: se2_types.Velocity | None = None
        node.create_timer(1 / 10, self._update_human_velocity)

    def get_latest_pose(
        self, target_frame: str, source_frame: str
    ) -> se2_types.Transform | None:
        """
        Returns the transform T_target_source (interpreted using left-handed
        matrix multiplication).
        """
        return self._cast(self._lookup_transform(target_frame, source_frame))

    def publish_2d_pose(
        self, target_frame: str, source_frame: str, pose: se2_types.Transform
    ) -> None:
        transform = TransformStamped()
        transform.header.frame_id = target_frame
        transform.header.stamp = self._node.get_clock().now().to_msg()
        transform.child_frame_id = source_frame

        transform.transform.translation.x = pose.x
        transform.transform.translation.y = pose.y
        transform.transform.rotation.z = math.sin(pose.theta / 2)
        transform.transform.rotation.w = math.cos(pose.theta / 2)

        self._tf2_pub.publish(TFMessage(transforms=[transform]))

    def publish_2d_poses(
        self, poses: list[tuple[str, str, se2_types.Transform]]
    ) -> None:
        ros_stamp = self._node.get_clock().now().to_msg()
        ros_transforms = []

        for target_frame, source_frame, pose in poses:
            transform = TransformStamped()
            transform.header.frame_id = target_frame
            transform.header.stamp = ros_stamp
            transform.child_frame_id = source_frame

            transform.transform.translation.x = pose.x
            transform.transform.translation.y = pose.y
            transform.transform.rotation.z = math.sin(pose.theta / 2)
            transform.transform.rotation.w = math.cos(pose.theta / 2)

            ros_transforms.append(transform)

        self._tf2_pub.publish(TFMessage(transforms=ros_transforms))

    def _lookup_transform(
        self, target_frame: str, source_frame: str
    ) -> TransformStamped | None:
        try:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1 / 100),  # type: ignore
            )
        except:  # noqa: E722
            return None

    @overload
    def _cast(self, ros_transform: None) -> None: ...

    @overload
    def _cast(self, ros_transform: TransformStamped) -> se2_types.Transform: ...

    def _cast(
        self, ros_transform: TransformStamped | None
    ) -> se2_types.Transform | None:
        if ros_transform is None:
            return None

        return se2_types.Transform(
            ros_transform.transform.translation.x,
            ros_transform.transform.translation.y,
            2.0
            * math.atan2(
                ros_transform.transform.rotation.z,
                ros_transform.transform.rotation.w,
            ),
        )

    def _update_human_velocity(self) -> None:
        if self._last_tf_map_human is None:
            self._last_tf_map_human = self._lookup_transform("map", "human_1")
            return

        current_tf_map_human = self._lookup_transform("map", "human_1")
        if current_tf_map_human is None:
            self._last_tf_map_human = None
            return

        odom = self._cast(self._last_tf_map_human).inverse() @ self._cast(
            current_tf_map_human
        )

        dt = _ros2_time_to_sec(current_tf_map_human.header.stamp) - _ros2_time_to_sec(
            self._last_tf_map_human.header.stamp
        )
        if dt == 0:
            return

        self._last_tf_map_human = current_tf_map_human
        self.latest_human_velocity = se2_types.Velocity(
            odom.x / dt,
            odom.y / dt,
            odom.theta / dt,
        )
