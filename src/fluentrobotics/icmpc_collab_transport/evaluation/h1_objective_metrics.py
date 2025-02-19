import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import tqdm

from fluentrobotics.icmpc_collab_transport import logger
from fluentrobotics.icmpc_collab_transport.core.se2_types import Transform, Vector

from . import association, parse_rosbags, stats


def calculate_success_rates() -> None:
    association_df = association.get_semilong_dataframe()

    success_rates = association_df.groupby("algorithm").mean(numeric_only=True)
    logger.info("Success Rates")
    print(success_rates.round(3))


def calculate_completion_times() -> None:
    association_df = association.get_semilong_dataframe()
    rosbag_lookup = parse_rosbags.get_filtered_rosbag_data_dict()

    # Pass/fail values in run_idx columns will be overwritten with durations
    duration_df = (
        association_df.copy()
        .drop(columns="success_rate")
        .astype(dict((f"run_{i}", "float64") for i in range(1, 4)))
    )
    for row in association_df.itertuples(index=True):
        bagfolder = getattr(row, "bagfolder")
        algorithm = getattr(row, "algorithm")

        for run_idx in range(1, 4):
            suceeded = getattr(row, f"run_{run_idx}")
            if not suceeded:
                duration_df.loc[row.Index, f"run_{run_idx}"] = math.nan
                continue
            ros_data = rosbag_lookup[(bagfolder, algorithm, run_idx)]
            duration = ros_data.tf_map_human_1[-1][0] - ros_data.tf_map_human_1[0][0]
            duration_df.loc[row.Index, f"run_{run_idx}"] = duration

    # Convert to a long dataframe and drop the run indices
    duration_df = duration_df.melt(
        id_vars=("response_id", "algorithm"),
        value_vars=("run_1", "run_2", "run_3"),
        value_name="time",
    )
    duration_df = duration_df.drop(columns=["variable"])

    stats.run_nonparametric_tests(duration_df, "time")


def calculate_accelerations() -> None:
    association_df = association.get_semilong_dataframe()
    rosbag_lookup = parse_rosbags.get_filtered_rosbag_data_dict()

    # Pass/fail values in run_idx columns will be overwritten with durations
    accel_df = (
        association_df.copy()
        .drop(columns="success_rate")
        .astype(dict((f"run_{i}", "float64") for i in range(1, 4)))
    )
    for row in tqdm.tqdm(
        association_df.itertuples(index=True),
        total=len(association_df),
    ):
        bagfolder = getattr(row, "bagfolder")
        algorithm = getattr(row, "algorithm")

        for run_idx in range(1, 4):
            suceeded = getattr(row, f"run_{run_idx}")
            if not suceeded:
                accel_df.loc[row.Index, f"run_{run_idx}"] = math.nan
                continue

            ros_data = rosbag_lookup[(bagfolder, algorithm, run_idx)]
            # Skip the first second, the last 4 seconds and downsample from 120 Hz to 10 Hz
            lskip = 120
            rskip = 4 * 120
            step = 12

            # Calculate three sets with slightly different left offsets to
            # mitigate random spikes caused by jitter.
            positions: list[list[tuple[float, Transform]]] = []
            for i in range(3):
                positions.append(
                    ros_data.tf_map_human_1[lskip + step * i // 3 : -rskip : step]
                )

            velocities: list[list[tuple[float, Vector]]] = []
            for position in positions:
                velocity = [
                    (
                        position[i][0],
                        (position[i][1].t - position[i - 1][1].t)
                        / (position[i][0] - position[i - 1][0]),
                    )
                    for i in range(1, len(position))
                ]
                velocities.append(velocity)

            # Use the velocity series that has the least worst outliers
            velocity = min(velocities, key=lambda v: max(x[1].norm for x in v))
            # Alternatively, we could take the median or mean of each index.

            acceleration: list[tuple[float, Vector]] = [
                (
                    velocity[i][0],
                    (velocity[i][1] - velocity[i - 1][1])
                    / (velocity[i][0] - velocity[i - 1][0]),
                )
                for i in range(1, len(velocity))
            ]
            accel_df.loc[row.Index, f"run_{run_idx}"] = np.mean(
                [x[1].norm for x in acceleration]
            )

    # Convert to a long dataframe and drop the run indices
    accel_df = accel_df.melt(
        id_vars=("response_id", "algorithm"),
        value_vars=("run_1", "run_2", "run_3"),
        value_name="acceleration",
    )
    accel_df = accel_df.drop(columns=["variable"])

    stats.run_nonparametric_tests(accel_df, "acceleration")


def main() -> None:
    fn_dispatch: dict[str, Callable[[], None]] = {
        "success": calculate_success_rates,
        "time": calculate_completion_times,
        "acceleration": calculate_accelerations,
    }

    argparser = argparse.ArgumentParser(Path(__file__).stem)
    argparser.add_argument("function", choices=fn_dispatch.keys())

    args = argparser.parse_args()
    fn_dispatch[args.function]()


if __name__ == "__main__":
    main()
