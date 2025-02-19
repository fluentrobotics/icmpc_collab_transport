import numba.np.extensions
import numpy as np
from numba import float64, njit
from numba.experimental import jitclass


@njit(cache=True)
def _construct(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    value = np.empty((2, 1), dtype=np.float64)
    value[0, 0] = x
    value[1, 0] = y
    return value


@njit(cache=True)
def normalized(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


@njit(cache=True)
def angle_from_to(v_from: np.ndarray, v_to: np.ndarray) -> float:
    a = normalized(v_from.flatten())
    b = normalized(v_to.flatten())

    return np.arctan2(numba.np.extensions.cross2d(a, b), np.dot(a, b)).item()


# @jitclass([("value", float64[:, :])])
class Vector:
    """R^2 Vector. Can also be used to represent a Point."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        _move_from: np.ndarray | None = None,
    ) -> None:
        if _move_from is not None:
            self.value = _move_from
        else:
            self.value = _construct(x, y)

    @property
    def x(self) -> float:
        return self.value[0].item()

    @x.setter
    def x(self, new_value: float) -> None:
        self.value[0] = new_value

    @property
    def y(self) -> float:
        return self.value[1].item()

    @y.setter
    def y(self, new_value: float) -> None:
        self.value[1] = new_value

    @property
    def norm(self) -> float:
        return np.linalg.norm(self.value).item()

    def normalized(self) -> "Vector":
        return self / self.norm

    def angle_to(self, other: "Vector") -> float:
        """Return the signed angle in [-pi, pi] from this Vector to the other
        Vector.

        Assumes both vectors have nonzero norm.
        """
        return angle_from_to(self.value, other.value)

    def dot(self, other: "Vector") -> float:
        return (self.value.T @ other.value).item()

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(_move_from=self.value + other.value)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(_move_from=self.value - other.value)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(_move_from=self.value * scalar)

    def __truediv__(self, scalar: float) -> "Vector":
        return Vector(_move_from=self.value / scalar)

    def __str__(self) -> str:
        return f"Vector[{self.x:6.3f} {self.y:6.3f}]"
