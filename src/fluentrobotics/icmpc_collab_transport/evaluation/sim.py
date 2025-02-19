import csv
import functools
import math
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fluentrobotics.icmpc_collab_transport import logger
from fluentrobotics.icmpc_collab_transport.core.passing_inference import (
    ActionLikelihoodDistribution,
    PriorDistribution,
)
from fluentrobotics.icmpc_collab_transport.core.se2_types import Transform, Vector
from fluentrobotics.icmpc_collab_transport.core.strategy import PassingStrategy


@functools.cache
def get_passing_labels() -> dict[int, list[PassingStrategy]]:
    def char_to_strategy(c: str) -> PassingStrategy:
        match c:
            case "L":
                return PassingStrategy.LEFT
            case "R":
                return PassingStrategy.RIGHT
            case _:
                raise ValueError()

    label_file = Path("data/eleyng/train_passing_labels.csv")
    labels: dict[int, list[PassingStrategy]] = defaultdict(list)
    with open(label_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            assert len(row) == 2
            run_id = int(row[0])
            labels[run_id] = [char_to_strategy(c) for c in row[1]]

    return labels


class DemonstrationData:
    def __init__(self, trajectory_file: Path, map_config_file: Path) -> None:
        self.run_id = int(trajectory_file.parent.stem)

        self.path: list[Transform] = []
        self.actions: list[Vector] = []
        self.goal: Transform = Transform()
        self.obstacles: list[Transform] = []

        self.passing_labels: list[PassingStrategy] = get_passing_labels()[self.run_id]

        #
        # Below is messy data wrangling
        #

        trajectory_data = dict(np.load(trajectory_file, allow_pickle=True))
        map_config = dict(np.load(map_config_file, allow_pickle=True))

        path: np.ndarray = trajectory_data["states"][:, :2]
        assert path.ndim == 2 and path.shape[1] == 2
        for state_idx in range(path.shape[0]):
            self.path.append(Transform(path[state_idx, 0], path[state_idx, 1]))

        actions: np.ndarray = trajectory_data["actions"]
        assert actions.ndim == 2 and actions.shape[1] == 4
        for action_idx in range(actions.shape[0]):
            action_sum = actions[action_idx, :2] + actions[action_idx, 2:]
            self.actions.append(Vector(action_sum[0], action_sum[1]))

        self.goal.x = map_config["goal"].item()["goal"][0]
        self.goal.y = map_config["goal"].item()["goal"][1]

        obstacles = map_config["obstacles"].item()["obstacles"]
        assert obstacles.ndim == 2 and obstacles.shape[1] == 2
        for obs_idx in range(obstacles.shape[0]):
            self.obstacles.append(
                Transform(obstacles[obs_idx, 0], obstacles[obs_idx, 1])
            )
        self.obstacles.sort(key=lambda tf: math.atan2(tf.y, tf.x))

        assert len(self.obstacles) == len(
            self.passing_labels
        ), "len(obstacles) does not match len(passing_labels)"


class ActionStatistics:
    def __init__(self) -> None:
        self.lookup_table: dict[
            str, np.ndarray[Literal["12, 24, 8"], np.dtype[np.float32]]
        ] = defaultdict(lambda: np.zeros((12, 24, 8), dtype=np.float32))

    def bin(self, state: Transform) -> tuple[int, int]:
        # Dataset uses x: [0, 1200], y: [0, 600]
        return int(state.x / 50), int(state.y / 50)

    def get_key(self, obstacle: Transform, goal: Transform) -> str:
        return f"{obstacle.x:.0f}-{obstacle.y:.0f}-{goal.x:.0f}-{goal.y:.0f}"

    def update_distribution(self, key: str, state: Transform, action: Vector) -> None:
        x, y = self.bin(state)

        angle = math.atan2(action.y, action.x)
        if angle < 0:
            angle += 2 * math.pi
        angle_idx = round(angle / (2 * math.pi) * 8) % 8

        self.lookup_table[key][y, x][angle_idx] += 1

    def get_distribution(
        self, key: str, state: Transform
    ) -> np.ndarray[Literal["8"], np.dtype[np.float32]] | None:
        x, y = self.bin(state)

        raw_count = self.lookup_table[key][y, x]
        total_count = raw_count.sum()
        if total_count == 0:
            return None
        else:
            return raw_count


@functools.cache
def get_train_dataset() -> list[DemonstrationData]:
    invalid_ids = set([101, 102, 103, 104, 105, 108, 110, 111, 115])

    dataset = []

    dataset_dir = Path("data/eleyng/demonstrations/trajectories/train")
    for child in dataset_dir.iterdir():
        if int(child.stem) in invalid_ids:
            continue

        trajectory_file = child / f"ep_{child.stem}.npz"
        map_config_file = Path(
            f"data/eleyng/demonstrations/map_cfg/ep_{child.stem}.npz"
        )

        data = DemonstrationData(trajectory_file, map_config_file)
        dataset.append(data)

    return dataset


def evaluate_prior() -> None:
    dataset = get_train_dataset()
    dataset = [datum for datum in dataset if len(datum.obstacles) == 1]

    Result = namedtuple("Result", ["run_id", "obs_idx", "t_pct", "theta", "is_correct"])
    results: list[Result] = []

    for datum in dataset:
        for obs_idx, (obstacle, label) in enumerate(
            zip(datum.obstacles, datum.passing_labels)
        ):
            # for t_pct in range(0, 101):
            #     state_idx = (len(datum.path) - 1) * t_pct / 100
            #     state_idx = round(state_idx)
            #     state = datum.path[state_idx]

            #     prior_dist = PriorDistribution(obstacle, datum.goal, state)
            #     theta = prior_dist._compute_theta(state)
            #     theta = round(theta * 180 / math.pi)
            #     is_correct = prior_dist.get_mode() == label
            #     results.append(Result(datum.run_id, obs_idx, t_pct, theta, is_correct))

            for state_idx in range(len(datum.path)):
                state = datum.path[state_idx]

                prior_dist = PriorDistribution(obstacle, datum.goal, state)
                t_pct = round(state_idx / len(datum.path) * 100)
                theta = prior_dist._compute_theta(state)
                theta = round(theta * 180 / math.pi)
                is_correct = prior_dist.get_mode() == label
                results.append(Result(datum.run_id, obs_idx, t_pct, theta, is_correct))

    df = pd.DataFrame.from_records(results, columns=Result._fields)
    logger.info("Prior Results:")
    logger.info(f"Accuracy: {df['is_correct'].mean():.3f}")

    fig, axs = plt.subplots(1, 2, sharey=True, layout="constrained", figsize=(8, 4))
    sns.lineplot(df, x="t_pct", y="is_correct", ax=axs[0])
    sns.lineplot(df, x="theta", y="is_correct", ax=axs[1])
    # sns.histplot(df, x="t_pct", y="theta", ax=axs[1])
    # sns.kdeplot(df, x="t_pct", y="theta", levels=4, ax=axs[1])

    plt.savefig("sim_prior.png", dpi=300)
    plt.close(fig)


def evaluate_action_likelihood() -> None:
    dataset = get_train_dataset()
    dataset = [datum for datum in dataset if len(datum.obstacles) == 1]

    action_stats = ActionStatistics()
    for datum in dataset:
        for obstacle in datum.obstacles:
            key = action_stats.get_key(obstacle, datum.goal)
            for state, action in zip(datum.path, datum.actions):
                action_stats.update_distribution(key, state, action)

    Result = namedtuple(
        "Result", ["run_id", "obs_idx", "t_pct", "theta", "count", "is_correct"]
    )
    results: list[Result] = []
    n_insufficient_data = 0

    for datum in dataset:
        for obs_idx, (obstacle, label) in enumerate(
            zip(datum.obstacles, datum.passing_labels)
        ):
            visited_bins: set[tuple[int, int]] = set()

            for state_idx in range(len(datum.path)):
                state = datum.path[state_idx]
                bin = action_stats.bin(state)

                # First two bins are not reliable
                # if len(visited_bins) < 2:
                #     visited_bins.add(bin)

                # if bin in visited_bins:
                #     continue
                # visited_bins.add(bin)
                prior_dist = PriorDistribution(obstacle, datum.goal, state)
                action_likelihood_dist = ActionLikelihoodDistribution(
                    obstacle, datum.goal, state
                )
                if label == PassingStrategy.LEFT:
                    est_dist = action_likelihood_dist.pmf[1]
                else:
                    est_dist = action_likelihood_dist.pmf[2]

                key = action_stats.get_key(obstacle, datum.goal)
                data_dist = action_stats.get_distribution(key, state)

                if data_dist is None:
                    n_insufficient_data += 1
                    continue

                t_pct = round(state_idx / len(datum.path) * 100)
                theta = prior_dist._compute_theta(state)
                theta = round(theta * 180 / math.pi)
                count = data_dist.sum()
                # count = np.count_nonzero(data_dist)
                is_correct = np.argmax(data_dist) == np.argmax(est_dist)
                results.append(
                    Result(datum.run_id, obs_idx, t_pct, theta, count, is_correct)
                )

    df = pd.DataFrame.from_records(results, columns=Result._fields)
    logger.info("Action Likelihood Results:")
    logger.info(f"Accuracy: {df['is_correct'].mean():.3f}")
    logger.info(
        f"Accuracy (t_pct>10): {df.loc[df['t_pct'] > 10, 'is_correct'].mean():.3f}"
    )
    logger.info(
        f"Accuracy (t_pct>20): {df.loc[df['t_pct'] > 20, 'is_correct'].mean():.3f}"
    )
    logger.info(
        f"Accuracy (t_pct>30): {df.loc[df['t_pct'] > 30, 'is_correct'].mean():.3f}"
    )

    fig, axs = plt.subplots(1, 2, sharey=False, layout="constrained", figsize=(8, 4))
    sns.lineplot(df, x="t_pct", y="is_correct", ax=axs[0])
    # sns.lineplot(df, x="count", y="is_correct", ax=axs[1])
    # sns.lineplot(df, x="count", y="is_correct", ax=axs[1])
    # sns.histplot(df, x="count", ax=axs[1])
    sns.scatterplot(df, x="t_pct", y="count", ax=axs[1])
    plt.savefig("sim_action.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    evaluate_prior()
    evaluate_action_likelihood()
