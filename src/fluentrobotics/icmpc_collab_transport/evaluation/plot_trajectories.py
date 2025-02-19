import bisect
import itertools
from collections import namedtuple
from pathlib import Path
from typing import TypeVar

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from fluentrobotics.icmpc_collab_transport.core.passing_inference import (
    ActionLikelihoodDistribution,
    PosteriorDistribution,
    PriorDistribution,
)
from fluentrobotics.icmpc_collab_transport.core.se2_types import Transform

from . import association, parse_rosbags

T_ = TypeVar("T_")


def find_nearest(a: list[tuple[float, T_]], v: float) -> tuple[float, T_]:
    idx = bisect.bisect(a, v, key=lambda x: x[0])

    if idx > 0 and (idx == len(a) or abs(v - a[idx - 1][0]) < (v - a[idx][0])):
        return a[idx - 1]
    else:
        return a[idx]


def plot_aggregate_heatmap() -> None:
    rosbag_lookup = parse_rosbags.get_filtered_rosbag_data_dict()
    association_df = association.get_semilong_dataframe()

    scale = 15  # pixels per meter
    h = 3 * scale
    w = 6 * scale
    state_visitation = {
        "IC-MPC": np.zeros((h, w), dtype=np.int64),
        "Vanilla-MPC": np.zeros((h, w), dtype=np.int64),
        "VRNN": np.zeros((h, w), dtype=np.int64),
    }

    for bagfolder, algorithm, run_idx in itertools.product(
        association_df["bagfolder"].unique(),
        association_df["algorithm"].unique(),
        [1, 2, 3],
    ):
        ros_data = rosbag_lookup[(bagfolder, algorithm, run_idx)]

        position = [
            (
                x[0],
                (find_nearest(ros_data.tf_map_base_link, x[0])[1] @ x[1]),
            )
            for x in ros_data.tf_base_link_object_center
        ]

        xs = [t[1].x for t in position]
        ys = [t[1].y for t in position]

        states_visited = np.zeros((h, w), dtype=np.int64)
        for x, y in zip(xs, ys):
            y2 = int((2 - x) * scale)
            x2 = int((3 - y) * scale)
            if 0 <= x2 < w and 0 <= y2 < h:
                states_visited[y2, x2] = 1

        state_visitation[algorithm] += states_visited

    axs: list[plt.Axes]
    fig, axs = plt.subplots(1, 3, layout="constrained", figsize=(7.5, 1.45))

    for i, algo in zip(range(3), state_visitation.keys()):
        im = axs[i].imshow(state_visitation[algo], cmap="magma")
        axs[i].set_title(algo, fontsize=10)
        axs[i].axis("off")
        # if i == 2:
        #     fig.colorbar(im)

    plt.savefig("heatmap.svg", dpi=600)


def plot_aggregate_entropy() -> None:
    rosbag_lookup = parse_rosbags.get_filtered_rosbag_data_dict()

    Record = namedtuple("Record", ["algorithm", "time", "entropy"])
    records: list[Record] = []

    # hardcode for now, because it's the same for all data
    obstacle = Transform(0.5, 0, 0)
    goal = Transform(0.5, -2.5, 0)
    for ros_data in rosbag_lookup.values():
        t_0 = ros_data.tf_base_link_object_center[0][0]

        for t, tf_base_link_object_center in ros_data.tf_base_link_object_center:
            tf_map_base_link = find_nearest(ros_data.tf_map_base_link, t)[1]
            tf_map_object_center = tf_map_base_link @ tf_base_link_object_center
            tf_map_human = find_nearest(ros_data.tf_map_human_1, t)[1]

            a_robot = find_nearest(ros_data.robot_vel, t)[1]
            a_human = find_nearest(ros_data.human_vel, t)[1]
            # convert to map coordinates
            a_robot = tf_map_base_link @ a_robot
            a_human = tf_map_human @ a_human

            prior = PriorDistribution(obstacle, goal, tf_map_object_center)
            action_likelihood = ActionLikelihoodDistribution(
                obstacle, goal, tf_map_object_center
            )
            posterior = PosteriorDistribution(
                prior, action_likelihood, a_human, a_robot
            )

            entropy = posterior.get_entropy()
            records.append(
                Record(
                    ros_data.algorithm,
                    round(t - t_0, ndigits=1),
                    entropy,
                )
            )

    df = pd.DataFrame.from_records(records, columns=Record._fields)
    df = df.rename(columns={"algorithm": "Algorithm"})

    fig, ax = plt.subplots(layout="constrained", figsize=(3.5, 2.25))
    sns.lineplot(
        df[df["time"] < 20],
        x="time",
        y="entropy",
        hue="Algorithm",
        hue_order=["IC-MPC", "Vanilla-MPC", "VRNN"],
        errorbar=None,
        palette="Set2",
        ax=ax,
    )
    # ax.set_ylabel(r"$H[P(\psi \mid \alpha, s, c)]$")
    ax.set_ylabel(" ")
    ax.set_xlabel("Time (s)")
    sns.despine(ax=ax)
    # plt.show()
    plt.savefig("entropy.svg", dpi=600)


def plot_entropy_individual(ros_data: parse_rosbags.RosbagData, ax: Axes) -> None:
    # hardcode for now, because it's the same for all data
    obstacle = Transform(0.5, 0, 0)
    goal = Transform(0.5, -2.5, 0)

    timestamps = []
    entropies = []

    t_0 = ros_data.tf_base_link_object_center[0][0]
    for t, tf_base_link_object_center in ros_data.tf_base_link_object_center:
        tf_map_base_link = find_nearest(ros_data.tf_map_base_link, t)[1]
        tf_map_object_center = tf_map_base_link @ tf_base_link_object_center
        tf_map_human = find_nearest(ros_data.tf_map_human_1, t)[1]

        a_robot = find_nearest(ros_data.robot_vel, t)[1]
        a_human = find_nearest(ros_data.human_vel, t)[1]
        # convert to map coordinates
        a_robot = tf_map_base_link @ a_robot
        a_human = tf_map_human @ a_human

        prior = PriorDistribution(obstacle, goal, tf_map_object_center)
        action_likelihood = ActionLikelihoodDistribution(
            obstacle, goal, tf_map_object_center
        )
        posterior = PosteriorDistribution(prior, action_likelihood, a_human, a_robot)

        entropy = posterior.get_entropy()
        timestamps.append(t - t_0)
        entropies.append(entropy)

    ax.plot(timestamps, entropies)
    ax.set_xlim(-1, 20)
    ax.set_ylim(-0.1, 1.1)


def plot_individual_trajectories() -> None:
    rosbag_lookup = parse_rosbags.get_filtered_rosbag_data_dict()
    association_df = association.get_semilong_dataframe()

    output_directory = Path("data/fluentrobotics/trajectories")
    output_directory.mkdir(exist_ok=True)

    for bagfolder, algorithm, run_idx in tqdm.tqdm(
        itertools.product(
            association_df["bagfolder"].unique(),
            association_df["algorithm"].unique(),
            [1, 2, 3],
        ),
        total=len(association_df["bagfolder"].unique())
        * len(association_df["algorithm"].unique())
        * 3,
        dynamic_ncols=True,
    ):
        suceeded = association_df.loc[
            (association_df["bagfolder"] == bagfolder)
            & (association_df["algorithm"] == algorithm),
            f"run_{run_idx}",
        ].item()

        fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(8, 4))
        ax = axs[0]

        ros_data = rosbag_lookup[(bagfolder, algorithm, run_idx)]

        ####
        obj_position = [
            (
                x[0],
                (find_nearest(ros_data.tf_map_base_link, x[0])[1] @ x[1]),
            )
            for x in ros_data.tf_base_link_object_center
        ]
        xs = np.array([t[1].x for t in obj_position])
        ys = np.array([t[1].y for t in obj_position])
        mpl_xs = ys
        mpl_ys = xs

        ax.plot(
            mpl_xs,
            mpl_ys,
            c=("tab:green" if suceeded else "tab:orange"),
        )

        ####
        robot_position = ros_data.tf_map_base_link
        xs = np.array([t[1].x for t in robot_position])
        ys = np.array([t[1].y for t in robot_position])
        mpl_xs = ys
        mpl_ys = xs
        ax.plot(
            mpl_xs,
            mpl_ys,
            c="gray",
        )

        ####
        human_position = ros_data.tf_map_human_1
        xs = np.array([t[1].x for t in human_position])
        ys = np.array([t[1].y for t in human_position])
        mpl_xs = ys
        mpl_ys = xs
        ax.plot(
            mpl_xs,
            mpl_ys,
            c="mediumpurple",
        )

        ####
        ax.scatter(0, 0.5, s=50, marker="s", c="tab:red")
        ax.scatter(-2.5, 0.5, s=50, marker="x", c="tab:blue")

        ax.axes.set_aspect("equal")
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-1.5, 2.5)
        ax.invert_xaxis()
        ax.set_ylabel("x")
        ax.set_xlabel("y")

        plot_entropy_individual(ros_data, axs[1])

        plt.savefig(
            output_directory / f"{algorithm}-{run_idx}-{bagfolder}.png", dpi=150
        )
        plt.close(fig)


def animate_all_trajectories(algorithm: str = "IC-MPC") -> None:
    rosbag_lookup = parse_rosbags.get_filtered_rosbag_data_dict()
    association_df = association.get_semilong_dataframe()

    output_directory = Path("data/fluentrobotics/trajectories")
    output_directory.mkdir(exist_ok=True)

    trajectories: list[list[tuple[float, Transform]]] = []
    succeeded = []
    for bagfolder, run_idx in itertools.product(
        association_df["bagfolder"].unique(),
        [1, 2, 3],
    ):
        rosbag_data = rosbag_lookup[(bagfolder, algorithm, run_idx)]
        tf_map_object_base_link = [
            (
                x[0],
                (find_nearest(rosbag_data.tf_map_base_link, x[0])[1] @ x[1]),
            )
            for x in rosbag_data.tf_base_link_object_center
        ]
        trajectories.append(tf_map_object_base_link)

        success = association_df.loc[
            (association_df["bagfolder"] == bagfolder)
            & (association_df["algorithm"] == algorithm),
            f"run_{run_idx}",
        ].item()
        succeeded.append(success)

    fps = 60
    dt = 1 / fps
    # Rows are frames (3s x 30fps)
    # Each column is a separate trajectory
    x_buffer = np.full((3 * fps, 72), np.nan)
    y_buffer = np.full((3 * fps, 72), np.nan)
    T = 40
    dt = T / (3 * fps)

    plt.style.use("dark_background")
    plt.rcParams["axes.facecolor"] = "#000000"
    plt.rcParams["figure.facecolor"] = "#000000"
    plt.rcParams["savefig.facecolor"] = "#000000"

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(6, 4))
    ax.axes.set_aspect("equal")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.5, 2.5)
    ax.invert_xaxis()

    lines = ax.plot(y_buffer, x_buffer)
    for line, success in zip(lines, succeeded):
        # https://davidmathlogic.com/colorblind/#%23005AB5-%23DC3220
        if success:
            line.set_color("#005ab5")
            line.set_alpha(0.8)
        else:
            line.set_color("#dc3220")

    pbar = tqdm.tqdm(total=x_buffer.shape[0])

    def update(frame_idx: int) -> list[Line2D]:
        for line_idx in range(len(lines)):
            trajectory = trajectories[line_idx]
            transform = find_nearest(trajectory, trajectory[0][0] + frame_idx * dt)
            x_buffer[frame_idx, line_idx] = transform[1].x
            y_buffer[frame_idx, line_idx] = transform[1].y

            lines[line_idx].set_xdata(y_buffer[:, line_idx])
            lines[line_idx].set_ydata(x_buffer[:, line_idx])

        pbar.update(1)
        return lines

    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=x_buffer.shape[0], interval=1 / fps
    )
    ffwriter = animation.FFMpegWriter(fps)
    anim.save(output_directory / f"{algorithm}.mp4", ffwriter, dpi=300)
    pbar.close()


def main() -> None:
    font = {
        "family": "Nimbus Roman",
        # "family": "Arial",
        "size": 8,
    }
    plt.rc("font", **font)
    plot_aggregate_heatmap()
    # plot_aggregate_entropy()
    # plot_individual_trajectories()
    # animate_all_trajectories("IC-MPC")
    # animate_all_trajectories("VRNN")
    # animate_all_trajectories("Vanilla-MPC")


if __name__ == "__main__":
    main()
