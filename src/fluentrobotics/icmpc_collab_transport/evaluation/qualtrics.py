from pathlib import Path

import pandas as pd

from fluentrobotics.icmpc_collab_transport import logger

from . import association
from .questionnaires import (
    FLUENCY_LABELS,
    ROSAS_COMPETENCE_LABELS,
    ROSAS_DISCOMFORT_LABELS,
    ROSAS_LABELS,
    ROSAS_WARMTH_LABELS,
)


class QualtricsData:
    def __init__(
        self,
        association_csv_file: Path = Path("data/fluentrobotics/data-association.csv"),
        qualtrics_csv_file: Path = Path("data/fluentrobotics/qualtrics.csv"),
    ) -> None:
        self.association_df = association.get_wide_dataframe(association_csv_file)
        self._qualtrics_df = self._read_qualtrics_df(qualtrics_csv_file)

        self.rosas_wide_df = self._get_rosas_wide_df(
            self.association_df, self._qualtrics_df
        )
        self.rosas_long_df = self._get_rosas_long_df(self.rosas_wide_df)
        self.fluency_wide_df = self._get_fluency_wide_df(
            self.association_df, self._qualtrics_df
        )
        self.fluency_long_df = self._get_fluency_long_df(self.fluency_wide_df)

        self.comment_long_df = self._get_comment_long_df(
            self.association_df, self._qualtrics_df
        )

    def print_participant_familiarity_stats(self) -> None:
        print("Familiarity with robotics technology:")
        print("\t1 = Not at all familiar")
        print("\t5 = Very familiar")
        print(f"\tMean: {self._qualtrics_df['Q4_1'].mean():.3f}")
        print(f"\tStddev: {self._qualtrics_df['Q4_1'].std():.3f}")

    @staticmethod
    def _read_qualtrics_df(qualtrics_csv_file: Path) -> pd.DataFrame:
        df = pd.read_csv(qualtrics_csv_file, skiprows=[1, 2], comment="#")
        # Columns required for parsing the Qualtrics export, but not relevant for
        # analysis.
        df = df.drop(
            columns=[
                "StartDate",
                "EndDate",
                "Status",
                "IPAddress",
                "Progress",
                "Duration (in seconds)",
                "Finished",
                "RecordedDate",
                "RecipientLastName",
                "RecipientFirstName",
                "RecipientEmail",
                "ExternalReference",
                "LocationLatitude",
                "LocationLongitude",
                "DistributionChannel",
                "UserLanguage",
            ],
            errors="ignore",
        )
        return df

    @staticmethod
    def _get_rosas_wide_df(
        association_df: pd.DataFrame, qualtrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        output_rows = []

        for association_row in association_df.itertuples(index=False):
            response_id = getattr(association_row, "response_id")
            qualtrics_row: pd.DataFrame = qualtrics_df[
                qualtrics_df["ResponseId"] == response_id
            ]
            if qualtrics_row.shape[0] == 0:
                logger.warning(f"No data found for ResponseId {response_id}")
                continue
            assert qualtrics_row.shape[0] == 1

            for set_idx in range(1, 4):
                algo_name = getattr(association_row, f"set{set_idx}algo")
                tmp_df = qualtrics_row.filter(like=f"{set_idx}_Q1_")
                tmp_df.columns = ROSAS_LABELS  # type: ignore
                tmp_df.insert(loc=0, column="response_id", value=response_id)
                tmp_df.insert(loc=1, column="algorithm", value=algo_name)
                output_rows.append(tmp_df)

        rosas_df = pd.concat(output_rows, ignore_index=True)

        rosas_df.insert(
            len(rosas_df.columns),
            column="warmth (mean)",
            value=rosas_df.loc[:, rosas_df.columns.isin(ROSAS_WARMTH_LABELS)].mean(
                axis=1
            ),
        )
        rosas_df.insert(
            len(rosas_df.columns),
            column="competence (mean)",
            value=rosas_df.loc[:, rosas_df.columns.isin(ROSAS_COMPETENCE_LABELS)].mean(
                axis=1
            ),
        )
        rosas_df.insert(
            len(rosas_df.columns),
            column="discomfort (mean)",
            value=rosas_df.loc[:, rosas_df.columns.isin(ROSAS_DISCOMFORT_LABELS)].mean(
                axis=1
            ),
        )

        return rosas_df

    @staticmethod
    def _get_rosas_long_df(rosas_wide_df: pd.DataFrame) -> pd.DataFrame:
        rosas_long_df = rosas_wide_df.melt(
            id_vars=["response_id", "algorithm"],
            value_vars=ROSAS_LABELS
            + ("warmth (mean)", "competence (mean)", "discomfort (mean)"),
            var_name="attribute",
            value_name="rating",
        )

        # Add subscale labels for seaborn catplot
        rosas_long_df.insert(loc=3, column="subscale", value="")
        rosas_long_df.loc[
            rosas_long_df["attribute"].isin(ROSAS_WARMTH_LABELS + ("warmth (mean)",)),
            "subscale",
        ] = "warmth"
        rosas_long_df.loc[
            rosas_long_df["attribute"].isin(
                ROSAS_COMPETENCE_LABELS + ("competence (mean)",)
            ),
            "subscale",
        ] = "competence"
        rosas_long_df.loc[
            rosas_long_df["attribute"].isin(
                ROSAS_DISCOMFORT_LABELS + ("discomfort (mean)",)
            ),
            "subscale",
        ] = "discomfort"

        return rosas_long_df

    @staticmethod
    def _get_fluency_wide_df(
        association_df: pd.DataFrame, qualtrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        output_rows = []

        for association_row in association_df.itertuples(index=False):
            response_id = getattr(association_row, "response_id")
            qualtrics_row: pd.DataFrame = qualtrics_df[
                qualtrics_df["ResponseId"] == response_id
            ]
            if qualtrics_row.shape[0] == 0:
                logger.warning(f"No data found for ResponseId {response_id}")
                continue
            assert qualtrics_row.shape[0] == 1

            for set_idx in range(1, 4):
                algo_name = getattr(association_row, f"set{set_idx}algo")
                tmp_df = qualtrics_row.filter(like=f"{set_idx}_Q2_")
                tmp_df.columns = FLUENCY_LABELS  # type: ignore
                tmp_df.insert(loc=0, column="response_id", value=response_id)
                tmp_df.insert(loc=1, column="algorithm", value=algo_name)
                output_rows.append(tmp_df)

        df = pd.concat(output_rows, ignore_index=True)
        df2 = df.copy(deep=True)
        df2.loc[:, df2.columns.str.endswith("(R)")] *= -1
        df2.loc[:, df2.columns.str.endswith("(R)")] += 8
        df.insert(
            len(df.columns),
            column="fluency (mean)",
            value=df2.loc[:, df2.columns.isin(FLUENCY_LABELS)].mean(axis=1),
        )
        return df

    @staticmethod
    def _get_fluency_long_df(fluency_wide_df: pd.DataFrame) -> pd.DataFrame:
        return fluency_wide_df.melt(
            id_vars=["response_id", "algorithm"],
            value_vars=FLUENCY_LABELS + ("fluency (mean)",),
            var_name="statement",
            value_name="rating",
        )

    @staticmethod
    def _get_comment_long_df(
        association_df: pd.DataFrame, qualtrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        output_rows = []

        for association_row in association_df.itertuples(index=False):
            response_id = getattr(association_row, "response_id")
            qualtrics_row: pd.DataFrame = qualtrics_df[
                qualtrics_df["ResponseId"] == response_id
            ]
            if qualtrics_row.shape[0] == 0:
                logger.warning(f"No data found for ResponseId {response_id}")
                continue
            assert qualtrics_row.shape[0] == 1

            for set_idx in range(1, 4):
                algo_name = getattr(association_row, f"set{set_idx}algo")
                tmp_df = qualtrics_row.filter(like=f"{set_idx}_Q4")
                tmp_df.columns = ["Comment"]  # type: ignore
                tmp_df.insert(loc=0, column="response_id", value=response_id)
                tmp_df.insert(loc=1, column="algorithm", value=algo_name)
                output_rows.append(tmp_df)

        return pd.concat(output_rows)


def print_comments() -> None:
    d = QualtricsData()

    comment_wide_df = d.comment_long_df.pivot(
        index="response_id", columns=["algorithm"]
    )

    for index, row in d.association_df.iterrows():
        row = row.to_frame().T
        print(
            row[
                ["response_id", "bagfolder", "set1pass", "set2pass", "set3pass"]
            ].to_string(header=False)
        )
        response_id = row["response_id"]

        for set_idx in range(1, 4):
            algorithm = row[f"set{set_idx}algo"].item()
            print(
                algorithm, comment_wide_df["Comment"].loc[response_id, algorithm].item()
            )
        print()

    # Group by algorithm
    # for algorithm in sorted(d.comment_long_df["algorithm"].unique()):
    #     df2 = d.comment_long_df[d.comment_long_df["algorithm"] == algorithm]
    #     print(f"============ {algorithm}")
    #     comments = df2["Comment"].to_list()
    #     for c in comments:
    #         if isinstance(c, str):
    #             print(c)
    #             print()
