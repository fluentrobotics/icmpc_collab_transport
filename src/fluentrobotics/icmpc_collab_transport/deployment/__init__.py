import sys

from fluentrobotics.icmpc_collab_transport import logger

try:
    import rclpy
except ImportError:
    logger.critical(
        f"You must be in a ROS 2 environment to use this package. ({__name__})"
    )
    sys.exit(1)
