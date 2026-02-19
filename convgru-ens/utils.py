import numpy as np

def rainrate_to_reflectivity(rainrate: np.ndarray) -> np.ndarray:
    """Convert rain rate to reflectivity using Marshall-Palmer relationship."""
    epsilon = 1e-16
    # We return 0 for any rain lighter than ~0.037mm/h
    return (10 * np.log10(200 * rainrate ** 1.6 + epsilon)).clip(0, 60)

def normalize_reflectivity(reflectivity: np.ndarray) -> np.ndarray:
    """Normalize reflectivity from [0, 60] to [-1, 1]."""
    return (reflectivity / 30.0) - 1.0

def denormalize_reflectivity(normalized: np.ndarray) -> np.ndarray:
    """Denormalize from [-1, 1] back to [0, 60] reflectivity."""
    return (normalized + 1.0) * 30.0

def reflectivity_to_rainrate(reflectivity: np.ndarray) -> np.ndarray:
    """Convert reflectivity back to rain rate (inverse Marshall-Palmer)."""
    # Z = 200 * R^1.6
    # R = (Z / 200)^(1/1.6)
    z_linear = 10 ** (reflectivity / 10.0)
    return (z_linear / 200.0) ** (1.0 / 1.6)

def rainrate_to_normalized(rainrate: np.ndarray) -> np.ndarray:
    """Convert rain rate directly to normalized reflectivity."""
    reflectivity = rainrate_to_reflectivity(rainrate)
    return normalize_reflectivity(reflectivity)

def normalized_to_rainrate(normalized: np.ndarray) -> np.ndarray:
    """Convert normalized reflectivity back to rain rate."""
    reflectivity = denormalize_reflectivity(normalized)
    return reflectivity_to_rainrate(reflectivity)