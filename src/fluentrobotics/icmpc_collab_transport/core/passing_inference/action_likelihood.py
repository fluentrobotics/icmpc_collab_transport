import math
from typing import Literal

import numpy as np
from numba import njit

from .. import se2_types
from ..se2_types import (
    Rotation,
    Transform,
    Vector,
    Velocity,
)
from ..se2_types.vector import angle_from_to, normalized
from ..strategy import PassingStrategy


@njit(cache=True)
def softmax(a: np.ndarray) -> np.ndarray:
    assert a.ndim == 1
    s = np.exp(a)
    s /= s.sum()
    return s


class ActionLikelihoodDistribution:
    # This cannot be too large, as the probability mass function will begin
    # to gain entropy.
    GRANULARITY_STEPS = 8

    def __init__(
        self,
        obstacle: Transform,
        goal: Transform,
        state: Transform,
    ) -> None:
        self._obstacle = obstacle
        self._goal = goal

        # self.pmf = self._compute_pmf(state)
        self.pmf = ActionLikelihoodDistribution._compute_pmf2(
            obstacle.value, goal.value, state.value, self.GRANULARITY_STEPS
        )

    @staticmethod
    @njit(cache=True)
    def _compute_pmf2(
        obstacle: np.ndarray,
        goal: np.ndarray,
        state: np.ndarray,
        GRANULARITY_STEPS: int,
    ) -> np.ndarray[tuple[Literal[3], Literal[9]], np.dtype[np.float64]]:
        """
        The return has 9 columns: 8 for cardinal directions, 9th is the zero
        vector (NO_ACTION)
        """
        distribution = np.zeros((3, GRANULARITY_STEPS + 1), dtype=np.float64)

        v_obs_state = state[:2, 2:] - obstacle[:2, 2:]
        v_obs_goal = goal[:2, 2:] - obstacle[:2, 2:]
        is_object_passed = abs(angle_from_to(v_obs_goal, v_obs_state)) < math.pi / 2

        v_state_goal = normalized(goal[:2, 2:] - state[:2, 2:])
        v_state_obs = normalized(obstacle[:2, 2:] - state[:2, 2:])

        theta_space = np.arange(float(GRANULARITY_STEPS)) / GRANULARITY_STEPS * math.tau
        for idx, theta in enumerate(theta_space):
            distribution[0, idx] = theta

            v_action_direction = se2_types.vector._construct(
                math.cos(theta), math.sin(theta)
            )

            if is_object_passed:
                distribution[1:, idx] = (v_state_goal.T @ v_action_direction).item()
            else:
                v_left_nominal = (
                    se2_types.rotation._construct(math.pi / 3) @ v_state_obs
                )
                v_right_nominal = (
                    se2_types.rotation._construct(-math.pi / 3) @ v_state_obs
                )

                # "Alignment" / dot product with nominal vectors.
                distribution[1, idx] = (v_left_nominal.T @ v_action_direction).item()
                distribution[2, idx] = (v_right_nominal.T @ v_action_direction).item()

                # Scaling to make "score" goal-directed.
                distribution[1:, idx] *= (
                    (v_action_direction.T @ v_state_goal).item() + 1
                ) / 2

        # NOTE(elvout): The last column represents the zero vector or "no
        # action". Prior to softmax, this value is initialized to zero (which is
        # consistent mathematically with the dot product calculations), but
        # afterwards may be conceptually odd to interpret as a probability. It
        # appears that the likelihood favors the zero vector for the opposite of
        # the maximum likelihood of the prior, which _could_ make sense (e.g.,
        # if the system is passing from the left and the human freezes, are they
        # considering changing the strategy? or maybe they're just pausing
        # because the robot is slow?).
        #
        # It's worth looking into whether there's a better way to set the
        # probability for the zero vector, but we'll first see what happens with
        # the above.
        distribution[0, -1] = math.nan

        # Normalize distributions to sum to 1. Since there are negative values,
        # we cannot use mean here.
        distribution[1] = softmax(distribution[1])
        distribution[2] = softmax(distribution[2])

        return distribution

    def _compute_pmf(
        self, state: Transform
    ) -> np.ndarray[tuple[Literal[3], Literal[9]], np.dtype[np.float64]]:
        """
        The return has 9 columns: 8 for cardinal directions, 9th is the zero
        vector (NO_ACTION)
        """
        distribution = np.zeros((3, self.GRANULARITY_STEPS + 1), dtype=np.float64)

        v_obs_state = state.t - self._obstacle.t
        v_obs_goal = self._goal.t - self._obstacle.t
        is_object_passed = abs(v_obs_goal.angle_to(v_obs_state)) < math.pi / 2

        v_state_goal = (self._goal.t - state.t).normalized()
        v_state_obs = (self._obstacle.t - state.t).normalized()

        theta_space = np.linspace(0, math.tau, self.GRANULARITY_STEPS, endpoint=False)
        for idx, theta in enumerate(theta_space):
            distribution[0, idx] = theta

            v_action_direction = Vector(math.cos(theta), math.sin(theta))

            if is_object_passed:
                distribution[1:, idx] = v_state_goal.dot(v_action_direction)
            else:
                v_left_nominal = Rotation(math.pi / 3) @ v_state_obs
                v_right_nominal = Rotation(-math.pi / 3) @ v_state_obs

                # "Alignment" / dot product with nominal vectors.
                distribution[1, idx] = v_left_nominal.dot(v_action_direction)
                distribution[2, idx] = v_right_nominal.dot(v_action_direction)

                # Scaling to make "score" goal-directed.
                distribution[1:, idx] *= (v_action_direction.dot(v_state_goal) + 1) / 2

        # NOTE(elvout): The last column represents the zero vector or "no
        # action". Prior to softmax, this value is initialized to zero (which is
        # consistent mathematically with the dot product calculations), but
        # afterwards may be conceptually odd to interpret as a probability. It
        # appears that the likelihood favors the zero vector for the opposite of
        # the maximum likelihood of the prior, which _could_ make sense (e.g.,
        # if the system is passing from the left and the human freezes, are they
        # considering changing the strategy? or maybe they're just pausing
        # because the robot is slow?).
        #
        # It's worth looking into whether there's a better way to set the
        # probability for the zero vector, but we'll first see what happens with
        # the above.
        distribution[0, -1] = math.nan

        # Normalize distributions to sum to 1. Since there are negative values,
        # we cannot use mean here.
        distribution[1] = softmax(distribution[1])
        distribution[2] = softmax(distribution[2])

        return distribution

    def __str__(self) -> str:
        return "\n".join(
            [
                "Action Likelihood:",
                " →  ↗   ↑  ↖   ←  ↙   ↓  ↘  NO_ACT",
                " ".join([f"{x:.3f}" for x in self.pmf[1]]),
                " ".join([f"{x:.3f}" for x in self.pmf[2]]),
            ]
        )

    def get_mode(self, strategy: PassingStrategy) -> Velocity:
        row = 1 if strategy == PassingStrategy.LEFT else 2
        argmax_idx = np.argmax(self.pmf[row])

        if argmax_idx == self.GRANULARITY_STEPS:
            return Velocity()
        else:
            theta = self.pmf[0, argmax_idx]
            return Velocity(math.cos(theta), math.sin(theta))

    def get_mode_m1(self, strategy: PassingStrategy) -> Velocity:
        """
        Get the second highest argmax, which may be helpful for visualization
        purposes.
        """
        row = 1 if strategy == PassingStrategy.LEFT else 2
        index = np.argsort(self.pmf[row])[-2]

        if index == self.GRANULARITY_STEPS:
            return Velocity()
        else:
            theta = self.pmf[0, index]
            return Velocity(math.cos(theta), math.sin(theta))
