import math

import numpy as np
import scipy.special  # type: ignore[import-untyped]

from ..se2_types import Vector, Velocity
from ..strategy import PassingStrategy
from .action_likelihood import ActionLikelihoodDistribution
from .prior import PriorDistribution


class PosteriorDistribution:
    def __init__(
        self,
        prior: PriorDistribution,
        action_likelihood: ActionLikelihoodDistribution,
        h_action: Velocity,
        r_action: Velocity,
    ) -> None:
        self._prior = prior
        self._action_likelihood = action_likelihood

        self.pmf = self._compute_pmf(h_action, r_action)

    def _discretize_action(self, action: Velocity) -> int:
        """Action is assumed to be [x, y, 0] instead of [x, 0, omega]."""

        if Vector(action.x, action.y).norm < 0.1:  # m / s
            return ActionLikelihoodDistribution.GRANULARITY_STEPS

        angle = math.atan2(action.y, action.x)
        if angle < 0:
            angle += math.tau
        idx = int(round(angle / math.tau * 8)) % 8
        return idx

    def _compute_pmf(
        self,
        h_action: Velocity,
        r_action: Velocity,
    ) -> dict[PassingStrategy, float]:
        h_action_idx = self._discretize_action(h_action)
        r_action_idx = self._discretize_action(r_action)

        p_l = (
            self._prior.pmf[PassingStrategy.LEFT]
            * self._action_likelihood.pmf[1, h_action_idx]
            * self._action_likelihood.pmf[1, r_action_idx]
        )
        p_r = (
            self._prior.pmf[PassingStrategy.RIGHT]
            * self._action_likelihood.pmf[2, h_action_idx]
            * self._action_likelihood.pmf[2, r_action_idx]
        )

        # NOTE(elvout): Normalize distribution to sum to 1. Since these are
        # already probabilities, we can use linear scaling.
        _sum = p_l + p_r
        p_l /= _sum
        p_r /= _sum

        return {
            PassingStrategy.LEFT: p_l,
            PassingStrategy.RIGHT: p_r,
        }

    def __str__(self) -> str:
        l = self.pmf[PassingStrategy.LEFT]
        r = self.pmf[PassingStrategy.RIGHT]

        return "\n".join(
            [
                "Posterior:",
                f"  p(LEFT  | state, actions) = {l:.3f}",
                f"  p(RIGHT | state, actions) = {r:.3f}",
            ]
        )

    def get_mode(self) -> PassingStrategy:
        return max(self.pmf, key=lambda k: self.pmf[k])

    def get_entropy(self) -> float:
        # 1 / ln(2) for log base conversion from e to 2
        RECIP_LN_2 = 1.4426950408889634

        dist_as_array = np.array(tuple(self.pmf.values()))
        # NOTE(elvout): scipy.stats.entropy is the intended user-friendly
        # function, but it has a fair amount of overhead (probably from array
        # conversion, normalization, case handling). Since we know we have a
        # valid probability distribution, we use the lower level function here
        # for a roughly 100x speedup (scipy 1.14.1, i7-13700).
        #
        # refs:
        # https://github.com/scipy/scipy/blob/92d2a85/scipy/stats/_entropy.py#L26
        # https://github.com/scipy/scipy/blob/92d2a85/scipy/special/_convex_analysis.pxd#L5
        return scipy.special.entr(dist_as_array).sum().item() * RECIP_LN_2

    def get_cost(self) -> float:
        return 2.0 * (1 - max(self.pmf.values()))
