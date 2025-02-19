import torch


def normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    """Normalize an angle to [-pi, pi]"""
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def kinematics(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    For pytorch_mppi.MPPI

    Input:
    s: robot global state  (shape: NUM_SAMPLES x 3)
    a: robot action   (shape: NUM_SAMPLES x 2)

    Output:
    next robot global state after executing action (shape: NUM_SAMPLES x 3)
    """
    assert s.ndim == 2 and s.shape[-1] == 3
    assert a.ndim == 2 and a.shape[-1] == 2

    dt = 0.25

    s2_ego = torch.zeros_like(s)
    d_theta = a[:, 1] * dt
    turning_radius = a[:, 0] / a[:, 1]

    s2_ego[:, 0] = torch.where(
        a[:, 1] == 0, a[:, 0] * dt, turning_radius * torch.sin(d_theta)
    )
    s2_ego[:, 1] = torch.where(
        a[:, 1] == 0, 0.0, turning_radius * (1.0 - torch.cos(d_theta))
    )
    s2_ego[:, 2] = torch.where(a[:, 1] == 0, 0.0, d_theta)

    s2_global = torch.zeros_like(s)
    s2_global[:, 0] = (
        s[:, 0] + s2_ego[:, 0] * torch.cos(s[:, 2]) - s2_ego[:, 1] * torch.sin(s[:, 2])
    )
    s2_global[:, 1] = (
        s[:, 1] + s2_ego[:, 0] * torch.sin(s[:, 2]) + s2_ego[:, 1] * torch.cos(s[:, 2])
    )
    s2_global[:, 2] = normalize_angle(s[:, 2] + s2_ego[:, 2])

    return s2_global
