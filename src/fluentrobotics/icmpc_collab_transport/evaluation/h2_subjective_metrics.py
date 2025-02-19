import argparse
from pathlib import Path
from typing import Callable

from pingouin import cronbach_alpha

from fluentrobotics.icmpc_collab_transport import logger

from . import qualtrics, stats
from .questionnaires import (
    FLUENCY_LABELS,
    ROSAS_COMPETENCE_LABELS,
    ROSAS_DISCOMFORT_LABELS,
    ROSAS_WARMTH_LABELS,
)


def evaluate_cronbach_alpha() -> None:
    qualtrics_data = qualtrics.QualtricsData()

    warmth_alpha = cronbach_alpha(
        qualtrics_data.rosas_wide_df.filter(items=ROSAS_WARMTH_LABELS)
    )
    competence_alpha = cronbach_alpha(
        qualtrics_data.rosas_wide_df.filter(items=ROSAS_COMPETENCE_LABELS)
    )
    discomfort_alpha = cronbach_alpha(
        qualtrics_data.rosas_wide_df.filter(items=ROSAS_DISCOMFORT_LABELS)
    )

    fluency_df = qualtrics_data.fluency_wide_df.copy(deep=True)
    # Reverse scale conversion: rating = 8 - reverse_rating
    fluency_df.loc[:, fluency_df.columns.str.endswith("(R)")] *= -1
    fluency_df.loc[:, fluency_df.columns.str.endswith("(R)")] += 8
    fluency_alpha = cronbach_alpha(fluency_df.filter(items=FLUENCY_LABELS))

    logger.info(f"(RoSAS/warmth)    : {warmth_alpha[0]:.2f}")
    logger.info(f"(RoSAS/competence): {competence_alpha[0]:.2f}")
    logger.info(f"(RoSAS/discomfort): {discomfort_alpha[0]:.2f}")
    logger.info(f"(Fluency)         : {fluency_alpha[0]:.2f}")


def run_tests(subscale: str) -> None:
    qualtrics_data = qualtrics.QualtricsData()

    if subscale.startswith("fluency"):
        stats.run_nonparametric_tests(qualtrics_data.fluency_wide_df, subscale)
    else:
        stats.run_nonparametric_tests(qualtrics_data.rosas_wide_df, subscale)


def main() -> None:
    fn_dispatch: dict[str, Callable[[], None]] = {
        "alpha": evaluate_cronbach_alpha,
        "warmth": lambda: run_tests("warmth (mean)"),
        "competence": lambda: run_tests("competence (mean)"),
        "discomfort": lambda: run_tests("discomfort (mean)"),
        "fluency": lambda: run_tests("fluency (mean)"),
    }

    argparser = argparse.ArgumentParser(Path(__file__).stem)
    argparser.add_argument("function", choices=fn_dispatch.keys())

    args = argparser.parse_args()
    fn_dispatch[args.function]()


if __name__ == "__main__":
    main()
