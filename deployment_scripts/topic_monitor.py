#! /usr/bin/env python3

import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Type

try:
    import rclpy
    import rclpy.qos
    import rclpy.serialization
    from builtin_interfaces.msg import Time
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from nav_msgs.msg import Odometry, Path
    from rclpy.node import Node
    from sensor_msgs.msg import (
        BatteryState,
        Imu,
        JointState,
    )
    from std_msgs.msg import Bool, Int32, String
    from tf2_msgs.msg import TFMessage
    from visualization_msgs.msg import MarkerArray
except ImportError:
    print(
        f"FATAL: You must be in a ROS 2 environment to run this script ({sys.argv[0]})"
    )
    sys.exit(1)

try:
    from audio_common_msgs.msg import AudioData
    from speech_recognition_msgs.msg import SpeechRecognitionCandidates
except ImportError:
    AudioData = None
    SpeechRecognitionCandidates = None


def human_readable_size(bytes: float) -> str:
    suffixes = ["B", "KB", "MB", "GB"]

    if bytes < 1:
        return "0 B"

    log1000 = int(math.log10(bytes) // 3)
    return f"{bytes / (1000 ** log1000):.1f} {suffixes[log1000]}"


class ANSI:
    PREFIX = "\x1b["

    # Color sequences
    RED = PREFIX + "31m"
    GREEN = PREFIX + "32m"
    YELLOW = PREFIX + "33m"
    CYAN = PREFIX + "36m"
    WHITE = PREFIX + "37m"

    # SGR sequences
    RESET = PREFIX + "0m"
    BOLD = PREFIX + "1m"
    BLINK = PREFIX + "5m"
    NO_BOLD = PREFIX + "22m"
    NO_BLINK = PREFIX + "25m"

    # Control Sequences
    PREV_LINE = PREFIX + "1F"
    CLEAR_LINE = PREFIX + "2K"

    @classmethod
    def rgb(cls, s: str, r: int, g: int, b: int) -> str:
        return f"{cls.PREFIX}38;2;{r};{g};{b}m {s}{cls.RESET}"

    @classmethod
    def rgb_f(cls, s: str, r: float, g: float, b: float) -> str:
        return cls.rgb(s, int(r * 255), int(g * 255), int(b * 255))


@dataclass
class TopicStatistics:
    # wall time of this machine
    recv_times: deque[float] = field(default_factory=deque)
    sizes_bytes: deque[int] = field(default_factory=deque)
    total_size_bytes: int = field(default=0)
    expected_frequency: float = field(default=0.0)

    def append(self, recv_time: float, size: int) -> None:
        self.recv_times.append(recv_time)
        self.sizes_bytes.append(size)
        self.total_size_bytes += size

    def prune(self, max_msg_age: float = 1.0) -> None:
        oldest_time = time.time() - max_msg_age

        while self.recv_times and self.recv_times[0] < oldest_time:
            self.recv_times.popleft()
            self.total_size_bytes -= self.sizes_bytes.popleft()


class TopicFrequencyNode(Node):
    def __init__(self) -> None:
        super().__init__("topic_frequency_monitor")

        # TODO(elvout): read topics and expected frequencies from config file
        topics: list[tuple[str, Type, float]] = [
            ("/battery", BatteryState, 30),
            ("/diagnostics", DiagnosticArray, 1),
            ("/imu_mobile_base", Imu, 30),
            ("/imu_wrist", Imu, 30),
            ("/is_runstopped", Bool, 30),
            ("/odom", Odometry, 30),
            ("/robot_description", String, 0),
            ("/stretch/joint_states", JointState, 30),
            #
            # tf: 30 Hz from Stretch, 120 Hz from mocap, 15 Hz from nodes
            # publishing human velocity data
            ("/tf", TFMessage, 30 + 120 + 15),
            ("/tf_static", TFMessage, 0.0),
            #
            ("/audio", AudioData, 16.0),
            ("/is_speeching", Bool, 0.0),
            ("/sound_direction", Int32, 0.0),
            ("/sound_localization", PoseStamped, 0.0),
            ("/speech_audio", AudioData, 0.0),
            ("/speech_to_text", SpeechRecognitionCandidates, 0.0),
            #
            ("/stretch/cmd_vel", TwistStamped, 15),
            ("/human_1/twist", TwistStamped, 15),
            ("/mppi/vis/rollouts", MarkerArray, 15),
            ("/vrnn_mppi/vis/path_object_center", Path, 15),
            ("/vrnn_mppi/vis/path_robot", Path, 15),
            ("/passing_strategy/pmf", String, 15),
        ]
        self.topic_stats: dict[str, TopicStatistics] = {}

        for topic_name, msg_type, expected_frequency in topics:
            if msg_type is None:
                continue

            self.topic_stats[topic_name] = TopicStatistics(
                expected_frequency=expected_frequency
            )
            self.create_subscription(
                msg_type,
                topic_name,
                self.make_callback(topic_name),
                # In bandwidth-constrained network environments, the bandwidth
                # of the network may be less than a single message. Thus, we
                # used BEST_EFFORT instead of RELIABLE, otherwise no messages
                # will make it through the network.
                # 10,
                rclpy.qos.QoSProfile(
                    depth=10,
                    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                    reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                ),
                raw=True,
            )

        # print(
        #     rclpy.qos.QoSProfile(
        #         **rclpy.impl.implementation_singleton.rclpy_implementation.rmw_qos_profile_t.predefined(
        #             "qos_profile_default"
        #         ).to_dict()
        #     )
        # )
        # print("----")
        # print(rclpy.qos.qos_profile_system_default)
        # print("----")
        # print(rclpy.qos.qos_profile_sensor_data)
        # print("----")

        self.create_timer(1, self.print_stats)

    def make_callback(self, topic_name: str) -> Callable[[bytes], None]:
        """
        rclpy won't let us use the same function reference as callbacks for
        multiple topics, so this creates a new function object but with the same
        logic.
        """

        def callback(raw_msg: bytes) -> None:
            self.topic_stats[topic_name].append(time.time(), len(raw_msg))

        return callback

    def print_stats(self) -> None:
        max_msg_age = 3.0
        total_bandwidth = 1e-2
        max_bandwidth = 1e-2

        for topic_name, stats in self.topic_stats.items():
            stats.prune(max_msg_age)

            total_bandwidth += stats.total_size_bytes
            max_bandwidth = max(max_bandwidth, stats.total_size_bytes)

        for topic_name, stats in self.topic_stats.items():
            bandwidth = stats.total_size_bytes
            intensity = 0.5 + 0.5 * math.sqrt(bandwidth / max_bandwidth)

            bandwidth_str = f"{human_readable_size(bandwidth / max_msg_age):>8s}"
            bandwidth_str = f"{ANSI.rgb_f(bandwidth_str, 0.0, intensity, 1.0)}/s"

            frequency = len(stats.recv_times) / max_msg_age
            if frequency >= 0.95 * stats.expected_frequency:
                frequency_color = ANSI.GREEN
            elif frequency >= 0.8 * stats.expected_frequency:
                frequency_color = ANSI.YELLOW
            else:
                frequency_color = ANSI.RED
            frequency_str = f"{frequency_color}{frequency:6.1f} msg/s{ANSI.RESET}"

            print(
                f"{topic_name:45s}| {len(stats.recv_times):4d} msgs | "
                # f"{len(stats.recv_times)/max_msg_age:6.1f} msg/s | "
                f"{frequency_str} | "
                f"{bandwidth_str}"
                # f"{ANSI.rgb_f(bandwidth_s, 0.0, green, 1.0)}/s"
            )

        print(
            f"{ANSI.BOLD}{human_readable_size(total_bandwidth / max_msg_age):>83s}/s{ANSI.RESET}"
        )

        for _ in range(len(self.topic_stats) + 1):
            print(ANSI.PREV_LINE, end="")


if __name__ == "__main__":
    rclpy.init(args=sys.argv)
    node = TopicFrequencyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
