import math
from typing import overload

import numpy as np
from numba import float64, njit
from numba.experimental import jitclass

from .rotation import Rotation
from .vector import Vector
from .velocity import Velocity


@njit(cache=True)
def _construct(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> np.ndarray:
    cos = math.cos(theta)
    sin = math.sin(theta)
    value = np.zeros((3, 3), dtype=np.float64)
    value[0, 0] = cos
    value[0, 1] = -sin
    value[0, 2] = x
    value[1, 0] = sin
    value[1, 1] = cos
    value[1, 2] = y
    value[2, 2] = 1.0
    return value


# @jitclass([("value", float64[:, :])])
class Transform:
    """SE(2) Transform"""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        _move_from: np.ndarray | None = None,
    ) -> None:
        if _move_from is not None:
            self.value = _move_from
        else:
            self.value = _construct(x, y, theta)

    @property
    def x(self) -> float:
        return self.value[0, 2].item()

    @x.setter
    def x(self, new_value: float) -> None:
        self.value[0, 2] = new_value

    @property
    def y(self) -> float:
        return self.value[1, 2].item()

    @y.setter
    def y(self, new_value: float) -> None:
        self.value[1, 2] = new_value

    @property
    def theta(self) -> float:
        return np.arctan2(self.value[1, 0], self.value[0, 0]).item()

    @theta.setter
    def theta(self, new_value: float) -> None:
        c = math.cos(new_value)
        s = math.sin(new_value)
        self.value[0, 0] = c
        self.value[0, 1] = -s
        self.value[1, 0] = s
        self.value[1, 1] = c

    @property
    def R(self) -> Rotation:
        """Rotation component as a matrix."""
        return Rotation(_move_from=self.value[:2, :2])

    @property
    def t(self) -> Vector:
        """Translation component as a column vector."""
        return Vector(_move_from=self.value[:2, 2:])

    @property
    def dist(self) -> float:
        """L2 norm of the translation component."""
        return self.t.norm

    def inverse(self) -> "Transform":
        return Transform(_move_from=np.linalg.inv(self.value))

    @overload
    def __matmul__(self, other: "Transform") -> "Transform": ...
    @overload
    def __matmul__(self, other: Vector) -> Vector: ...
    @overload
    def __matmul__(self, other: Velocity) -> Velocity: ...

    def __matmul__(
        self, other: "Transform | Vector | Velocity"
    ) -> "Transform | Vector | Velocity":
        if isinstance(other, Transform):
            return Transform(_move_from=self.value @ other.value)
        elif isinstance(other, Vector):
            return (self.R @ other) + self.t
        elif isinstance(other, Velocity):
            return self.R @ other
        else:
            raise TypeError()

    def __str__(self) -> str:
        return f"Transform[{self.x:6.3f} {self.y:6.3f} {self.theta:6.3f}]"
