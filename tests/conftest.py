import numpy as np
import pytest

from convgru_ensemble.lightning_model import RadarLightningModel


@pytest.fixture
def model_small():
    """Small model with num_blocks=2 for fast testing."""
    return RadarLightningModel(
        input_channels=1,
        num_blocks=2,
        forecast_steps=2,
        ensemble_size=2,
        noisy_decoder=False,
    )


@pytest.fixture
def sample_rain_rate():
    """Synthetic rain rate data of shape (T, H, W)."""
    rng = np.random.default_rng(42)
    return rng.random((4, 16, 16), dtype=np.float32) * 10.0  # 0-10 mm/h
