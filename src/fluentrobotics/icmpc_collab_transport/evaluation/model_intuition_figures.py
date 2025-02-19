import math

import matplotlib.pyplot as plt
import numpy as np

from fluentrobotics.icmpc_collab_transport.core.passing_inference import (
    ActionLikelihoodDistribution,
    PriorDistribution,
)
from fluentrobotics.icmpc_collab_transport.core.se2_types import (
    Rotation,
    Transform,
    Vector,
)
from fluentrobotics.icmpc_collab_transport.core.strategy import PassingStrategy


def prior_figure() -> None:
    scale = 151
    h = 3 * scale
    w = 6 * scale

    histogram = np.zeros((h, w), dtype=np.float64)
    #   Center = (0.5, 0)
    #   upper boundary: x = 2
    #   lower boundary: x = -1
    #   left boundary: y = 3
    #   right boundary: y = -3
    #
    #                       x
    #                       ^
    #                       |
    #                  y <---

    obstacle = Transform(0.5, 0)
    goal = Transform(0.5, -2.5)

    for r in range(h):
        for c in range(w):
            x = r / scale * -1 + 2
            y = c / scale * -1 + 3

            prior = PriorDistribution(obstacle, goal, Transform(x, y))
            histogram[r, c] = prior.pmf[PassingStrategy.LEFT]

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(3.5, 1.5))

    im = ax.imshow(histogram, cmap="magma")
    ax.axis("off")
    fig.colorbar(im)
    plt.savefig("prior.svg", dpi=600)
    plt.close(fig)


def action_likelihood_figure() -> None:
    X = np.linspace(-1, 2, 12)
    Y = np.linspace(-3, 3, 20)

    obstacle = Transform(0.5, 0)
    goal = Transform(0.5, -2.5)

    V = np.zeros((X.shape[0], Y.shape[0], 2), dtype=np.float64)
    for x_i, x in enumerate(X):
        for y_i, y in enumerate(Y):
            dist = ActionLikelihoodDistribution(obstacle, goal, Transform(x, y))
            pmf = dist.pmf[1]

            v1 = dist.get_mode(PassingStrategy.LEFT)
            v1 = Vector(v1.x, v1.y)
            # TODO(elvout): clarity; rotate because env axes are 90deg from
            # standard cartesian
            v1 = Rotation(math.pi / 2) @ v1

            v2 = dist.get_mode_m1(PassingStrategy.LEFT)
            v2 = Vector(v2.x, v2.y)
            v2 = Rotation(math.pi / 2) @ v2

            probs = np.sort(pmf)
            if v2.norm != 0:
                v3 = (v1 * probs[-1] + v2 * probs[-2]) / (probs[-1] + probs[-2])
            else:
                v3 = v1
            # v3 = v1
            # V[x_i, y_i] = v1.x, v1.y
            V[x_i, y_i] = v3.x, v3.y

    fig, ax = plt.subplots(layout="constrained")
    ax.quiver(Y, X, V[:, :, 0], V[:, :, 1])
    ax.set_aspect("equal")
    ax.invert_xaxis()
    plt.savefig("action.svg", dpi=600)


def main() -> None:
    prior_figure()
    action_likelihood_figure()


if __name__ == "__main__":
    main()
