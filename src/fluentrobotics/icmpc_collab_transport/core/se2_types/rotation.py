import math
from typing import overload

import numpy as np
from numba import float64, njit
from numba.experimental import jitclass

from .vector import Vector
from .velocity import Velocity


@njit(cache=True)
def _construct(theta: float = 0.0) -> np.ndarray:
    cos = math.cos(theta)
    sin = math.sin(theta)
    value = np.empty((2, 2), dtype=np.float64)
    value[0, 0] = cos
    value[0, 1] = -sin
    value[1, 0] = sin
    value[1, 1] = cos
    return value


# @jitclass([("value", float64[:, :])])
class Rotation:
    """SO(2) Rotation"""

    def __init__(
        self,
        theta: float = 0.0,
        _move_from: np.ndarray | None = None,
    ) -> None:
        if _move_from is not None:
            self.value = _move_from
        else:
            self.value = _construct(theta)

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

    def inverse(self) -> "Rotation":
        return Rotation(_move_from=self.value.T.copy())

    @overload
    def __matmul__(self, other: "Rotation") -> "Rotation": ...

    @overload
    def __matmul__(self, other: Vector) -> Vector: ...

    @overload
    def __matmul__(self, other: Velocity) -> Velocity: ...

    def __matmul__(
        self, other: "Rotation | Vector | Velocity"
    ) -> "Rotation | Vector | Velocity":
        if isinstance(other, Rotation):
            return Rotation(_move_from=self.value @ other.value)
        elif isinstance(other, Vector):
            return Vector(_move_from=self.value @ other.value)
        elif isinstance(other, Velocity):
            new_velocity = Velocity(omega=other.omega)
            new_velocity.value[0:2, :] = self.value @ other.value[0:2, :]
            return new_velocity
        else:
            raise TypeError()

    def __str__(self) -> str:
        return f"Rotation[{self.theta:6.3f}]"
