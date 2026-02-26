import numpy as np
import xarray as xr

from convgru_ensemble.lightning_model import RadarLightningModel


def test_inference_on_sample_data():
    """End-to-end inference on the sample NetCDF with a freshly initialized model."""
    ds = xr.open_dataset("examples/sample_data.nc")
    rain = ds["RR"].values  # (54, 1400, 1200)

    # Use 6 past frames, full spatial extent
    past = rain[:6].astype(np.float32)
    _, H, W = past.shape

    model = RadarLightningModel(
        input_channels=1,
        num_blocks=3,
        forecast_steps=4,
        ensemble_size=1,
        noisy_decoder=False,
    )

    preds = model.predict(past, forecast_steps=4, ensemble_size=1)

    assert preds.shape == (1, 4, H, W)
    assert np.isfinite(preds).all()
    assert preds.dtype == np.float64 or preds.dtype == np.float32
