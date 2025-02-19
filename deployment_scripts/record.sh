#! /usr/bin/env bash

set -e


if [[ "$ROS_VERSION" != "2" ]]; then
    echo "FATAL: You must be in a ROS 2 environment to run this script (${0})."
    exit 1
fi

if [[ "$ROS_DISTRO" != "humble" ]]; then
    echo "WARNING: This script (${0}) has only been tested in ROS 2 Humble."
    sleep 1
fi

if [[ "$(hostname)" != "stretch"* ]]; then
    echo "WARNING: This script (${0}) should run on the robot."
    sleep 1
fi

topics=(
    # Published by the Stretch ROS 2 Driver.
    /battery
    /diagnostics
    /imu_mobile_base
    /imu_wrist
    /is_runstopped
    /odom
    /robot_description
    /stretch/joint_states

    # Stretch + Motion Capture TF Trees
    /tf
    /tf_static

    # Stretch Microphone
    /audio
    /is_speeching
    /sound_direction
    /sound_localization
    /speech_audio
    /speech_to_text

    # Stretch Commanded Velocity
    /stretch/cmd_vel

    # Human velocity derived from tf data
    /human_1/twist

    # MPPI / path stuff
    /mppi/vis/rollouts
    /vrnn_mppi/vis/path_object_center
    /vrnn_mppi/vis/path_robot

    /passing_strategy/pmf
)

ros2 bag record "${topics[@]}"
