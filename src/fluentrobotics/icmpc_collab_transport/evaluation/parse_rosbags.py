import bisect
import copy
import functools
import gc
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import tqdm
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from rosbags.highlevel import AnyReader
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

from fluentrobotics.icmpc_collab_transport import logger
from fluentrobotics.icmpc_collab_transport.core.se2_types import Transform, Velocity

from . import parallel


class AnyReaderWrapper(AnyReader):
    def __init__(self, bagpath: Path) -> None:
        super().__init__([bagpath])

    def all_topic_messages(self, topic: str) -> tuple[list[float], list[Any]]:
        topic_connections = [c for c in self.connections if c.topic == topic]

        timestamps: list[float] = []
        messages: list[Any] = []
        if not topic_connections:
            return timestamps, messages

        for connection, timestamp_ns, rawdata in self.messages(
            connections=topic_connections
        ):
            timestamps.append(timestamp_ns * 1e-9)
            messages.append(self.deserialize(rawdata, connection.msgtype))

        assert len(timestamps) == len(messages)
        return timestamps, messages

    def parse_time(self, msg: Time) -> float:
        return float(msg.sec) + float(msg.nanosec) * 1e-9

    def parse_tf_msgs(
        self, ros_tf_msgs: list[TFMessage], parent: str, child: str
    ) -> list[tuple[float, Transform]]:
        transforms: list[tuple[float, Transform]] = []
        for tf_msg in ros_tf_msgs:
            ros_transform_msg: TransformStamped
            for ros_transform_msg in tf_msg.transforms:
                if (
                    ros_transform_msg.child_frame_id == child
                    and ros_transform_msg.header.frame_id == parent
                ):
                    stamp = self.parse_time(ros_transform_msg.header.stamp)
                    tf = Transform(
                        x=ros_transform_msg.transform.translation.x,
                        y=ros_transform_msg.transform.translation.y,
                        theta=(
                            2.0
                            * math.atan2(
                                ros_transform_msg.transform.rotation.z,
                                ros_transform_msg.transform.rotation.w,
                            )
                        ),
                    )
                    transforms.append((stamp, tf))

        # Fix jitter by overwriting pose jumps with the last valid pose. The
        # proper way to do this is probably interpolation, but detecting the
        # interval might be difficult.
        for i in range(1, len(transforms)):
            # Jitter appears to occur due to z-axis flips. Checking for large
            # angular displacements appears to work.
            #
            # TODO: instead look at the direction of the z-axis in the 3D
            # rotation matrix?
            if abs(transforms[i][1].theta - transforms[i - 1][1].theta) > 0.5:
                new_tf = transforms[i][1]
                new_tf.theta = transforms[i - 1][1].theta
                transforms[i] = (transforms[i][0], new_tf)

        return transforms

    def parse_bool_msgs(self, topic: str) -> list[tuple[float, bool]]:
        ros_bool_msgs: list[Bool]
        stamps, ros_bool_msgs = self.all_topic_messages(topic)

        return [(ts, msg.data) for (ts, msg) in zip(stamps, ros_bool_msgs)]

    def parse_twiststamped_msgs(self, topic: str) -> list[tuple[float, Velocity]]:
        ros_twiststamped_msgs: list[TwistStamped]
        _, ros_twiststamped_msgs = self.all_topic_messages(topic)
        return [
            (
                self.parse_time(msg.header.stamp),
                Velocity(
                    msg.twist.linear.x,
                    msg.twist.linear.y,
                    msg.twist.angular.z,
                ),
            )
            for msg in ros_twiststamped_msgs
        ]

    def parse_odometry_msgs(self, topic: str) -> list[tuple[float, Velocity]]:
        ros_odometry_msgs: list[Odometry]
        _, ros_odometry_msgs = self.all_topic_messages(topic)
        return [
            (
                self.parse_time(msg.header.stamp),
                Velocity(
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.angular.z,
                ),
            )
            for msg in ros_odometry_msgs
        ]

    def parse_odometry_msgs2(self, topic: str) -> list[tuple[float, Transform]]:
        """
        TODO: unused? parse odometry pose data
        """
        ros_odometry_msgs: list[Odometry]
        _, ros_odometry_msgs = self.all_topic_messages(topic)
        tf_0 = Transform(
            ros_odometry_msgs[0].pose.pose.position.x,
            ros_odometry_msgs[0].pose.pose.position.y,
            2.0
            * math.atan2(
                ros_odometry_msgs[0].pose.pose.orientation.z,
                ros_odometry_msgs[0].pose.pose.orientation.w,
            ),
        )
        return [
            (
                self.parse_time(msg.header.stamp),
                tf_0.inverse()
                @ Transform(
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    2.0
                    * math.atan2(
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w,
                    ),
                ),
            )
            for msg in ros_odometry_msgs
        ]

    def parse_vrnn_predictions(self, topic: str) -> list[tuple[float, list[Transform]]]:
        ros_path_msgs: list[PathMsg]
        _, ros_path_msgs = self.all_topic_messages(topic)

        return [
            (
                self.parse_time(msg.header.stamp),
                [
                    Transform(
                        x=pose.pose.position.x,
                        y=pose.pose.position.y,
                        theta=2.0
                        * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w),
                    )
                    for pose in msg.poses
                ],
            )
            for msg in ros_path_msgs
        ]


class RosbagData:
    def __init__(self, rosbag_path: Path | None) -> None:
        self.path: Path | None = None
        self.bagfolder: str | None = None
        self.algorithm: str | None = None
        self.run_idx: int | None = None

        self.tf_map_base_link: list[tuple[float, Transform]] = []
        self.tf_map_human_1: list[tuple[float, Transform]] = []
        self.tf_base_link_object_center: list[tuple[float, Transform]] = []

        self.is_runstopped: list[tuple[float, bool]] = []
        self.cmd_vel: list[tuple[float, Velocity]] = []
        self.robot_vel: list[tuple[float, Velocity]] = []
        self.human_vel: list[tuple[float, Velocity]] = []

        # Only populated if algorithm is VRNN.
        # Transforms are relative to the map frame.
        self.vrnn_predictions: list[tuple[float, list[Transform]]] | None = None

        ########

        if rosbag_path is not None:
            self.path = rosbag_path
            self.bagfolder = rosbag_path.parent.name
            with AnyReaderWrapper(rosbag_path) as reader:
                _, tf_msgs = reader.all_topic_messages("/tf")
                self.tf_map_base_link = reader.parse_tf_msgs(
                    tf_msgs, "map", "base_link"
                )
                self.tf_map_human_1 = reader.parse_tf_msgs(tf_msgs, "map", "human_1")
                self.tf_base_link_object_center = reader.parse_tf_msgs(
                    tf_msgs, "base_link", "object_center"
                )
                self.is_runstopped = reader.parse_bool_msgs("/is_runstopped")
                self.cmd_vel = reader.parse_twiststamped_msgs("/stretch/cmd_vel")
                self.robot_vel = reader.parse_odometry_msgs("/odom")
                self.human_vel = reader.parse_twiststamped_msgs("/human_1/twist")

                if "/vrnn_mppi/vis/path_object_center" in reader.topics:
                    self.algorithm = "VRNN"
                    self.vrnn_predictions = reader.parse_vrnn_predictions(
                        "/vrnn_mppi/vis/path_object_center"
                    )
                elif "/passing_strategy/pmf" in reader.topics:
                    self.algorithm = "IC-MPC"
                else:
                    self.algorithm = "Vanilla-MPC"

    def copy_range(self, left_ts: float, right_ts: float) -> "RosbagData":
        d = copy.copy(self)

        d.tf_map_base_link = [
            x for x in d.tf_map_base_link if left_ts <= x[0] <= right_ts
        ]
        d.tf_map_human_1 = [x for x in d.tf_map_human_1 if left_ts <= x[0] <= right_ts]
        d.tf_base_link_object_center = [
            x for x in d.tf_base_link_object_center if left_ts <= x[0] <= right_ts
        ]
        d.is_runstopped = [x for x in d.is_runstopped if left_ts <= x[0] <= right_ts]
        d.cmd_vel = [x for x in d.cmd_vel if left_ts <= x[0] <= right_ts]
        d.robot_vel = [x for x in d.robot_vel if left_ts <= x[0] <= right_ts]

        if d.vrnn_predictions is not None:
            d.vrnn_predictions = [
                x for x in d.vrnn_predictions if left_ts <= x[0] <= right_ts
            ]

        return d

    # @staticmethod
    # def dump(instance: "RosbagData") -> None:
    #     cache_path = instance.rosbag_path.parent / "cache" / instance.rosbag_path.name
    #     cache_path.parent.mkdir(exist_ok=True)

    #     with open(cache_path, "wb") as f:
    #         pickle.dump(instance, f, protocol=4)

    # @staticmethod
    # def load(rosbag_path: Path) -> "RosbagData":
    #     cache_path = rosbag_path.parent / "cache" / rosbag_path.name

    #     if not cache_path.exists():
    #         instance = RosbagData(rosbag_path)
    #         RosbagData.dump(instance)
    #         return instance
    #     else:
    #         with open(cache_path, "rb") as f:
    #             gc.disable()
    #             instance = pickle.load(f)
    #             gc.enable()
    #             return instance


def get_rosbag_paths() -> list[Path]:
    d = Path("data/fluentrobotics")
    return [p for p in d.rglob("rosbag2*") if p.is_dir()]


def read_rosbag_data() -> list[RosbagData]:
    rosbags = get_rosbag_paths()

    cache_path = Path("data/fluentrobotics/cache.pkl")
    rosbag_data: list[RosbagData] = []
    if cache_path.exists():
        logger.info("Reading data from cache file")
        with open(cache_path, "rb") as f:
            gc.disable()
            rosbag_data = pickle.load(f)
            gc.enable()

        if len(rosbags) != len(rosbag_data):
            logger.error(
                f"Cached data length mismatch. Expected {len(rosbags)}, got {len(rosbag_data)}"
            )
            rosbag_data.clear()
        elif dir(RosbagData(None)) != dir(rosbag_data[0]):
            logger.error("Cached data type is old (field mismatch)")
            rosbag_data.clear()
        else:
            logger.success("Cached data seems ok")

    if len(rosbag_data) == 0:
        logger.info("Reading data from rosbags")
        rosbag_data = parallel.fork_join(RosbagData, rosbags, n_proc=12)
        logger.info("Writing data to cache file")
        with open(cache_path, "wb") as f:
            pickle.dump(rosbag_data, f, protocol=4)

    algo_counter: Counter[str | None] = Counter()
    for d in rosbag_data:
        algo_counter[d.algorithm] += 1
    if not all([count == len(rosbags) // 3 for count in algo_counter.values()]):
        logger.warning(f"Algorithm parsing might not be working: {algo_counter}")

    return rosbag_data


def get_filtered_rosbag_data(debug: bool = False) -> list[RosbagData]:
    rosbags = get_rosbag_paths()

    filtered_rosbag_data: list[RosbagData] = []
    cache_path = Path("data/fluentrobotics/filtered-cache.pkl")

    if cache_path.exists():
        logger.info("Reading filtered data from cache file")
        with open(cache_path, "rb") as f:
            gc.disable()
            filtered_rosbag_data = pickle.load(f)
            gc.enable()

        if len(rosbags) * 3 != len(filtered_rosbag_data):
            logger.error(
                f"Cached filtered data length mismatch. Expected {len(rosbags) * 3}, got {len(filtered_rosbag_data)}"
            )
            filtered_rosbag_data.clear()
        elif dir(RosbagData(None)) != dir(filtered_rosbag_data[0]):
            logger.error("Cached data type is old (field mismatch)")
            filtered_rosbag_data.clear()
        else:
            logger.success("Cached filtered data seems ok")

    if len(filtered_rosbag_data) == 0:
        rosbag_data = read_rosbag_data()

        for d in tqdm.tqdm(rosbag_data):
            key = f"{d.bagfolder}-{d.algorithm}"

            fig, axs = plt.subplots(
                2, 1, sharex=True, layout="constrained", figsize=(5, 5)
            )

            axs[0].plot(
                [x[0] for x in d.is_runstopped], [x[1] for x in d.is_runstopped]
            )
            axs[0].scatter(
                [x[0] for x in d.cmd_vel],
                [(x[1].x + x[1].omega) != 0 for x in d.cmd_vel],
                s=1,
                c="tab:purple",
            )

            left_idx = 0
            for run_idx in range(1, 4):
                while left_idx < len(d.is_runstopped) and d.is_runstopped[left_idx][1]:
                    left_idx += 1
                right_idx = left_idx + 1
                while (
                    right_idx < len(d.is_runstopped)
                    and not d.is_runstopped[right_idx][1]
                ):
                    right_idx += 1

                if right_idx == len(d.is_runstopped):
                    right_idx -= 1

                left_ts = d.is_runstopped[left_idx][0]
                right_ts = d.is_runstopped[right_idx][0]

                # messy code for single edge case where the robot was de-runstopped
                # but it wasn't in navigation mode so it was re-runstopped
                if right_ts - left_ts < 3:
                    left_idx = right_idx
                    while (
                        left_idx < len(d.is_runstopped) and d.is_runstopped[left_idx][1]
                    ):
                        left_idx += 1
                    right_idx = left_idx + 1
                    while (
                        right_idx < len(d.is_runstopped)
                        and not d.is_runstopped[right_idx][1]
                    ):
                        right_idx += 1

                    if right_idx == len(d.is_runstopped):
                        right_idx -= 1

                    left_ts = d.is_runstopped[left_idx][0]
                    right_ts = d.is_runstopped[right_idx][0]

                right_idx2 = bisect.bisect(d.cmd_vel, right_ts, key=lambda x: x[0])
                right_ts = d.cmd_vel[right_idx2 - 1][0]

                right_idx3 = bisect.bisect(
                    d.is_runstopped, right_ts, key=lambda x: x[0]
                )

                axs[0].plot(
                    [x[0] for x in d.is_runstopped[left_idx:right_idx3]],
                    [0.5] * (right_idx3 - left_idx),
                    label=f"{run_idx}",
                )

                left_idx = right_idx

                d2 = d.copy_range(left_ts, right_ts)
                d2.run_idx = run_idx
                filtered_rosbag_data.append(d2)

            axs[0].legend()
            if debug:
                plt.savefig(f"{key}.png", dpi=300)
            plt.close(fig)

        logger.info("Writing filtered data to cache file")
        with open(cache_path, "wb") as f:
            pickle.dump(filtered_rosbag_data, f, protocol=4)

    return filtered_rosbag_data


@functools.cache
def get_filtered_rosbag_data_dict() -> dict[tuple[str, str, int], RosbagData]:
    """
    Returns map (bagfolder, algorithm, run_idx) -> data
    """
    rosbag_lookup: dict[tuple[str, str, int], RosbagData] = {}
    for d in get_filtered_rosbag_data():
        assert d.bagfolder is not None
        assert d.algorithm is not None
        assert d.run_idx is not None
        rosbag_lookup[(d.bagfolder, d.algorithm, d.run_idx)] = d

    return rosbag_lookup
