import copy
import math

import numpy as np

from fluentrobotics.icmpc_collab_transport import logger

from ..se2_types import Transform, Velocity
from ..strategy import PassingStrategy
from .action_likelihood import ActionLikelihoodDistribution
from .posterior import PosteriorDistribution
from .prior import PriorDistribution


class InferenceWrapper:
    def __init__(self, obstacle: Transform, goal: Transform) -> None:
        self._obstacle = obstacle
        self._goal = goal

        self._object_state_history: list[Transform] = []
        self._latest_h_action = Velocity()
        self._latest_r_action = Velocity()

    def add_object_state(self, state: Transform) -> None:
        # Only add this state if it exceeds a minimum displacement threshold to
        # avoid blowing up state history length.
        if (
            len(self._object_state_history) == 0
            or (self._object_state_history[-1].inverse() @ state).dist > 0.05
        ):
            self._object_state_history.append(state)

    def update_h_action(self, action: Velocity) -> None:
        self._latest_h_action = action

    def update_r_action(self, action: Velocity) -> None:
        self._latest_r_action = action

    def get_prior_str(self) -> str:
        insufficient_data_msg = "\n".join(
            [
                "Prior: (insufficient trajectory data)",
                "  p(LEFT  | state) = 0.5",
                "  p(RIGHT | state) = 0.5",
            ]
        )
        if len(self._object_state_history) == 0:
            return insufficient_data_msg

        return str(
            PriorDistribution(
                self._obstacle, self._goal, self._object_state_history[-1]
            )
        )

    def get_action_likelihood_str(self) -> str:
        insufficient_data_msg = "Action Likelihood: (insufficient trajectory data)"

        if len(self._object_state_history) == 0:
            return insufficient_data_msg

        return str(
            ActionLikelihoodDistribution(
                self._obstacle, self._goal, self._object_state_history[-1]
            )
        )

    def get_posterior_str(self) -> str:
        insufficient_data_msg = "\n".join(
            [
                "Posterior: (insufficient trajectory data)",
                "  p(LEFT  | state, actions) = 0.5",
                "  p(RIGHT | state, actions) = 0.5",
            ]
        )
        if len(self._object_state_history) == 0:
            return insufficient_data_msg

        # NOTE(elvout): this might be redundant
        prior = PriorDistribution(
            self._obstacle, self._goal, self._object_state_history[-1]
        )
        action_likelihood = ActionLikelihoodDistribution(
            self._obstacle, self._goal, self._object_state_history[-1]
        )

        self._posterior = PosteriorDistribution(
            prior,
            action_likelihood,
            self._latest_h_action,
            self._latest_r_action,
        )
        return str(self._posterior)

    def get_robot_action_minimizing_entropy(self) -> Velocity | None:
        if len(self._object_state_history) == 0:
            return None

        latest_state = self._object_state_history[-1]
        prior = PriorDistribution(self._obstacle, self._goal, latest_state)
        action_likelihood = ActionLikelihoodDistribution(
            self._obstacle, self._goal, latest_state
        )

        posterior = PosteriorDistribution(
            prior, action_likelihood, Velocity(), Velocity()
        )
        if posterior.get_entropy() == 0:
            # obstacle is passed already - action distributions are the same
            strategy = posterior.get_mode()
            return action_likelihood.get_mode(strategy)

        action_space: list[Velocity] = []
        for angle in np.linspace(0, math.tau, 8, endpoint=False):
            action_space.append(Velocity(math.cos(angle), math.sin(angle)))
        action_space.append(Velocity())

        min_expected_entropy = np.inf
        argmin_action = Velocity()

        H_array = np.zeros((len(action_space), len(action_space)), dtype=np.float64)

        # p(a | x_t), marginalize out the strategy
        marginal_action_likelihood = np.zeros(
            (ActionLikelihoodDistribution.GRANULARITY_STEPS + 1), dtype=np.float64
        )
        for a_idx, action in enumerate(action_space):
            marginal_action_likelihood[a_idx] = (
                action_likelihood.pmf[1, a_idx] * prior.pmf[PassingStrategy.LEFT]
                + action_likelihood.pmf[2, a_idx] * prior.pmf[PassingStrategy.RIGHT]
            )

        logger.info(f"{marginal_action_likelihood.sum()}, {marginal_action_likelihood}")

        for r_idx, r_action in enumerate(action_space):
            for h_idx, h_action in enumerate(action_space):
                if False and r_idx > h_idx:
                    # NOTE(elvout): This is here to speed up this function by a
                    # factor of ~2. This will only work if the posterior
                    # distribution uses the same action likelihood distribution
                    # for both agents.
                    H_array[r_idx, h_idx] = H_array[h_idx, r_idx]
                else:
                    H_array[r_idx, h_idx] = (
                        marginal_action_likelihood[h_idx]
                        * PosteriorDistribution(
                            prior, action_likelihood, h_action, r_action
                        ).get_entropy()
                    )

            H_mean = np.mean(H_array[r_idx])

            if H_mean < min_expected_entropy:
                min_expected_entropy = H_mean
                argmin_action = r_action

        return argmin_action

    def get_robot_action_minimizing_entropy_seeded(self) -> Velocity | None:
        if len(self._object_state_history) == 0:
            return None

        latest_state = self._object_state_history[-1]
        prior = PriorDistribution(self._obstacle, self._goal, latest_state)
        action_likelihood = ActionLikelihoodDistribution(
            self._obstacle, self._goal, latest_state
        )

        posterior = PosteriorDistribution(
            prior, action_likelihood, Velocity(), Velocity()
        )
        if posterior.get_entropy() == 0:
            # obstacle is passed already - action distributions are the same
            strategy = posterior.get_mode()
            return action_likelihood.get_mode(strategy)

        action_space: list[Velocity] = [Velocity()]
        for angle in np.linspace(0, math.tau, 8, endpoint=False):
            action_space.append(Velocity(math.cos(angle), math.sin(angle)))

        min_expected_entropy = np.inf
        argmin_action = Velocity()

        for r_action in action_space:
            H = PosteriorDistribution(
                prior, action_likelihood, self._latest_h_action, r_action
            ).get_entropy()

            if H < min_expected_entropy:
                min_expected_entropy = H
                argmin_action = r_action

        return argmin_action


class ItTakesTwoSimInferenceWrapper(InferenceWrapper):
    """
    This class exists to prevent polluting the It Takes Two simulator code base
    with too much logic.

    Handles conversions from left-handed image coordinate system to standard
    right-handed coordinate system.
    """

    def __init__(
        self,
        obstacle: Transform,
        goal: Transform,
        state_pixels_to_meter: float,
        action_pixels_to_meter: float,
    ) -> None:
        self._state_pixels_to_meter = state_pixels_to_meter
        self._action_pixels_to_meter = action_pixels_to_meter

        obstacle = copy.deepcopy(obstacle)
        goal = copy.deepcopy(goal)
        # NOTE: negating y from image coordinate system to right-handed system
        obstacle.x /= self._state_pixels_to_meter
        obstacle.y /= -self._state_pixels_to_meter
        goal.x /= self._state_pixels_to_meter
        goal.y /= -self._state_pixels_to_meter

        super().__init__(obstacle, goal)

    def add_object_state(self, x: float, y: float, theta: float) -> None:  # type: ignore
        # NOTE: negating y from image coordinate system to right-handed system
        state = Transform(
            x / self._state_pixels_to_meter, -y / self._state_pixels_to_meter, theta
        )
        super().add_object_state(state)

    def update_h_action(self, x: float, y: float) -> None:  # type: ignore
        # NOTE: negating y from image coordinate system to right-handed system
        action = Velocity(
            x / self._action_pixels_to_meter, -y / self._action_pixels_to_meter
        )
        super().update_h_action(action)

    def update_r_action(self, x: float, y: float) -> None:  # type: ignore
        # NOTE: negating y from image coordinate system to right-handed system
        action = Velocity(
            x / self._action_pixels_to_meter, -y / self._action_pixels_to_meter
        )
        super().update_r_action(action)

    def get_robot_action_minimizing_entropy(self) -> Velocity | None:
        action = super().get_robot_action_minimizing_entropy()

        if action is not None:
            # NOTE: negating y from right-handed system to image coordinate system
            # The action is rescaled from [-1, 1] in the simulator dynamics code.
            action.y = -action.y

        return action

    def get_robot_action_minimizing_entropy_seeded(self) -> Velocity | None:
        action = super().get_robot_action_minimizing_entropy_seeded()

        if action is not None:
            # NOTE: negating y from right-handed system to image coordinate system
            # The action is rescaled from [-1, 1] in the simulator dynamics code.
            action.y = -action.y

        return action
