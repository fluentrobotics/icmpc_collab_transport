class Bounds:
    REAL_X_MAX = 1.92
    REAL_X_MIN = -0.86
    REAL_Y_MAX = 2.71
    REAL_Y_MIN = -2.85

    REAL_X_RANGE = REAL_X_MAX - REAL_X_MIN
    REAL_Y_RANGE = REAL_Y_MAX - REAL_Y_MIN

    # sim x_min and y_min are both zero
    SIM_X_RANGE = 1200
    SIM_Y_RANGE = 600

    @classmethod
    def info(cls) -> None:
        print(f"Simulator env. aspect ratio: {cls.SIM_X_RANGE / cls.SIM_Y_RANGE:.2f}")
        print(f"Real env aspect ratio: {cls.REAL_Y_RANGE / cls.REAL_X_RANGE:.2f}")
        print(
            f"Simulator pixels per real meter: {cls.SIM_X_RANGE / cls.REAL_Y_RANGE:.2f}"
        )

if __name__ == "__main__":
    Bounds.info()
