import numpy as np


def apply_radiation_noise(
    hyperspectral_patch: np.ndarray,
    alpha_range=(0.9, 1.1),
    beta: float = 1 / 25,
) -> np.ndarray:
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=hyperspectral_patch.shape,
    )
    return alpha * hyperspectral_patch + beta * noise
