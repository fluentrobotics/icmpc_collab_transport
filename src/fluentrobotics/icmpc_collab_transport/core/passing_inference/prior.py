import math

import numpy as np
from numba import njit

from ..se2_types import Transform
from ..se2_types.vector import angle_from_to
from ..strategy import PassingStrategy


@njit(cache=True)
def compute_theta(obstacle: np.ndarray, goal: np.ndarray, state: np.ndarray) -> float:
    v_obs_state = state[:2, 2:] - obstacle[:2, 2:]
    v_obs_goal = goal[:2, 2:] - obstacle[:2, 2:]

    return angle_from_to(v_obs_goal, v_obs_state)


class PriorDistribution:
    def __init__(
        self,
        obstacle: Transform,
        goal: Transform,
        state: Transform | list[Transform],
    ) -> None:
        self._obstacle = obstacle
        self._goal = goal

        if isinstance(state, list):
            if len(state) == 0:
                raise ValueError()

            # Backtrack to the last state satisfying |theta| >= pi/2 to determine
            # which side the obstacle was passed by.
            state_history = state
            # If no states satisfy the condition, the object is already
            # passed when the path begins.
            state = state_history[0]
            for _state in reversed(state_history):
                if abs(self._compute_theta(_state)) >= math.pi / 2:
                    state = _state
                    break

        # self.pmf = self._compute_pmf(state)
        pmf_values = PriorDistribution._compute_pmf2(
            obstacle.value, goal.value, state.value
        )
        self.pmf = {
            PassingStrategy.LEFT: pmf_values[0],
            PassingStrategy.RIGHT: pmf_values[1],
        }

    @staticmethod
    @njit(cache=True)
    def _compute_pmf2(
        obstacle: np.ndarray, goal: np.ndarray, state: np.ndarray
    ) -> list[float]:
        theta = compute_theta(obstacle, goal, state)

        p = 0.5 + (math.pi - abs(theta)) / (math.pi)
        # TODO(elvout): properly handle when the table has passed the object
        p = min(p, 1.0)

        # NOTE: numba does not work with dicts
        if theta >= 0:
            return [p, 1.0 - p]
        else:
            return [1.0 - p, p]

    def _compute_theta(self, state: Transform) -> float:
        # v_obs_state = state.t - self._obstacle.t
        # v_obs_goal = self._goal.t - self._obstacle.t

        # return v_obs_goal.angle_to(v_obs_state)
        return compute_theta(self._obstacle.value, self._goal.value, state.value)

    def _compute_pmf(self, state: Transform) -> dict[PassingStrategy, float]:
        theta = self._compute_theta(state)

        p = 0.5 + (math.pi - abs(theta)) / (math.pi)
        # TODO(elvout): properly handle when the table has passed the object
        p = min(p, 1.0)

        if theta >= 0:
            return {
                PassingStrategy.LEFT: p,
                PassingStrategy.RIGHT: 1 - p,
            }
        else:
            return {
                PassingStrategy.LEFT: 1 - p,
                PassingStrategy.RIGHT: p,
            }

    def __str__(self) -> str:
        l = self.pmf[PassingStrategy.LEFT]
        r = self.pmf[PassingStrategy.RIGHT]

        return "\n".join(
            [
                "Prior:",
                f"  p(LEFT  | state) = {l:.3f}",
                f"  p(RIGHT | state) = {r:.3f}",
            ]
        )

    def get_mode(self) -> PassingStrategy:
        return max(self.pmf, key=lambda k: self.pmf[k])
