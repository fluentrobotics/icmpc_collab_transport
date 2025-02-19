import pandas as pd
import pingouin  # type: ignore

from fluentrobotics.icmpc_collab_transport import logger

SIGNIFICANCE_LEVEL = 0.05


def run_parametric_tests(data: pd.DataFrame, dv: str) -> None:
    logger.info(f"Parametric tests for '{dv}'")

    normality = pingouin.normality(
        data,
        dv=dv,
        group="algorithm",
    )
    normality_pass = normality["normal"].min()
    logger.log(
        "SUCCESS" if normality_pass else "ERROR",
        f"Normality: {'PASS' if normality_pass else 'FAIL'}",
    )
    print(normality)

    sphericity = pingouin.sphericity(
        data,
        dv=dv,
        within="algorithm",
        subject="response_id",
    )
    sphericity_pass = sphericity.spher
    logger.log(
        "SUCCESS" if sphericity_pass else "ERROR",
        f"Sphericity: {'PASS' if sphericity_pass else 'FAIL'}",
    )
    print(sphericity)

    homoscedasticity = pingouin.homoscedasticity(
        data,
        dv=dv,
        group="algorithm",
    )
    homoscedasticity_pass = homoscedasticity["equal_var"].min()
    logger.log(
        "SUCCESS" if homoscedasticity_pass else "ERROR",
        f"Homoscedasticity: {'PASS' if homoscedasticity_pass else 'FAIL'}",
    )
    print(homoscedasticity)

    logger.info("One-way RM ANOVA")
    anova = pingouin.rm_anova(
        data,
        dv=dv,
        within="algorithm",
        subject="response_id",
    )
    print(anova)

    logger.info("Paired t test")
    p = anova.loc[0, "p-unc"]
    if p > SIGNIFICANCE_LEVEL:
        logger.warning(f"VALID ONLY FOR MEAN AND STD (ANOVA p={p:.3f})")
    pairwise = pingouin.pairwise_tests(
        data=data,
        dv=dv,
        within="algorithm",
        subject="response_id",
        parametric=True,
        # https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1380484/
        padjust="holm",
        effsize="cohen",
        return_desc=True,
    )
    print(pairwise)


def run_nonparametric_tests(data: pd.DataFrame, dv: str) -> None:
    logger.info(f"Nonparametric tests for '{dv}'")

    logger.info("Friedman")
    # https://en.wikipedia.org/wiki/Kendall%27s_W
    # https://cran.r-project.org/web/packages/effectsize/vignettes/interpret.html#kendalls-coefficient-of-concordance
    friedman = pingouin.friedman(
        data,
        dv=dv,
        within="algorithm",
        subject="response_id",
        method="chisq",
    )
    friedman[["W", "Q"]] = friedman[["W", "Q"]].round(2)
    print(friedman)

    logger.info("Wilcoxon paired test")
    p = friedman.loc["Friedman", "p-unc"]
    if p > SIGNIFICANCE_LEVEL:
        logger.warning(f"VALID ONLY FOR MEAN AND STD (Friedman p={p:.3f})")
    # The W statistic of the Wilcoxon test may not match p values of reference tables (e.g., [1])
    # because 0 values are omitted, which decreases N.
    #
    # [1]: https://users.sussex.ac.uk/~grahamh/RM1web/WilcoxonHandoout2011.pdf
    pairwise = pingouin.pairwise_tests(
        data=data,
        dv=dv,
        within="algorithm",
        subject="response_id",
        parametric=False,
        # https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1380484/
        padjust="holm",
        effsize="cohen",
        return_desc=True,
    )
    pairwise = pairwise.drop(columns="p-unc")
    pairwise[["mean(A)", "mean(B)", "std(A)", "std(B)", "cohen"]] = pairwise[
        ["mean(A)", "mean(B)", "std(A)", "std(B)", "cohen"]
    ].round(2)
    print(pairwise)
