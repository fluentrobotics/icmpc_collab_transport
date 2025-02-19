from collections import namedtuple
from pathlib import Path

import pandas as pd


def get_wide_dataframe(
    path: Path = Path("data/fluentrobotics/data-association.csv"),
) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def get_semilong_dataframe(
    path: Path = Path("data/fluentrobotics/data-association.csv"),
) -> pd.DataFrame:
    """
    Returns a long-ish-form dataframe, obtained by splitting each row in the CSV
    (a true wide-form dataframe) into (num_algorithms = 3) separate rows for
    each algorithm.

    Reinterprets the binary pass/fail coding for each algorithm into a success
    rate and binary True/False status for each run index.
    """

    Record = namedtuple(
        "Record",
        [
            "response_id",
            "bagfolder",
            "algorithm",
            "success_rate",
            "run_1",
            "run_2",
            "run_3",
        ],
    )
    records = []

    for row in get_wide_dataframe(path).itertuples(index=False):
        response_id = getattr(row, "response_id")
        bagfolder = getattr(row, "bagfolder")

        for set_idx in range(1, 4):
            algo_name = getattr(row, f"set{set_idx}algo")
            algo_pass = getattr(row, f"set{set_idx}pass")

            success_rate = int(algo_pass, base=2).bit_count() / 3
            run_pass = [c == "1" for c in algo_pass[-3:]]
            records.append(
                Record(
                    response_id,
                    bagfolder,
                    algo_name,
                    success_rate,
                    run_pass[0],
                    run_pass[1],
                    run_pass[2],
                )
            )

    return pd.DataFrame.from_records(records, columns=Record._fields)


def print_algorithm_success_rates() -> None:
    df = get_semilong_dataframe()
    df = df.melt(
        id_vars=["algorithm", "bagfolder"],
        value_vars=[f"run_{i}" for i in range(1, 4)],
        var_name="run_number",
        value_name="pass",
    )
    df["run_number"] = df["run_number"].map(lambda s: int(s[-1]))

    for algorithm in sorted(df["algorithm"].unique()):
        success_rate = df.loc[df["algorithm"] == algorithm, "pass"].mean()
        print(f"{algorithm}: {success_rate:.3f}")
        for run_number in range(1, 4):
            run_success_rate = df.loc[
                (df["algorithm"] == algorithm) & (df["run_number"] == run_number),
                "pass",
            ].mean()
            print(f"Run {run_number}: {run_success_rate:.3f}")


if __name__ == "__main__":
    print_algorithm_success_rates()
