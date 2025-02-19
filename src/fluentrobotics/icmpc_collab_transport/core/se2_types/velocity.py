import numpy as np
from numba import float64, njit
from numba.experimental import jitclass


@njit(cache=True)
def _construct(x: float = 0.0, y: float = 0.0, omega: float = 0.0) -> np.ndarray:
    value = np.empty((3, 1), dtype=np.float64)
    value[0, 0] = x
    value[1, 0] = y
    value[2, 0] = omega
    return value


# @jitclass([("value", float64[:, :])])
class Velocity:
    """SE(2) Velocity"""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        omega: float = 0.0,
        _move_from: np.ndarray | None = None,
    ) -> None:
        if _move_from is not None:
            self.value = _move_from
        else:
            self.value = _construct(x, y, omega)

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
    def omega(self) -> float:
        return self.value[2].item()

    @omega.setter
    def omega(self, new_value: float) -> None:
        self.value[2] = new_value

    def __str__(self) -> str:
        return f"Velocity[{self.x:6.3f} {self.y:6.3f} {self.omega:6.3f}]"
