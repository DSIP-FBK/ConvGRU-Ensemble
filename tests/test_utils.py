import numpy as np

from convgru_ensemble.utils import normalized_to_rainrate, rainrate_to_normalized


def test_roundtrip_conversion():
    rain = np.array([0.0, 1.0, 5.0, 20.0, 50.0], dtype=np.float32)
    normalized = rainrate_to_normalized(rain)
    recovered = normalized_to_rainrate(normalized)
    np.testing.assert_allclose(recovered, rain, rtol=0.01, atol=0.05)


def test_zero_rain_maps_to_low_normalized():
    rain = np.array([0.0], dtype=np.float32)
    normalized = rainrate_to_normalized(rain)
    # Zero rain should map to approximately -1 (0 dBZ → normalized -1)
    assert normalized[0] < -0.9
