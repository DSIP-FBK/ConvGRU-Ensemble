import numpy as np
import torch
from lightning_model import RadarLightningModel


def test_predict_handles_unpadded_inputs():
    model = RadarLightningModel(
        input_channels=1,
        num_blocks=1,
        forecast_steps=2,
        ensemble_size=1,
        noisy_decoder=False,
    )
    past = np.zeros((4, 8, 8), dtype=np.float32)

    preds = model.predict(past, forecast_steps=2, ensemble_size=1)

    assert preds.shape == (1, 2, 8, 8)
    assert np.isfinite(preds).all()


def test_from_checkpoint_delegates_to_lightning_loader(monkeypatch):
    captured = {}

    def fake_loader(cls, checkpoint_path, map_location=None, strict=None):
        captured["checkpoint_path"] = checkpoint_path
        captured["map_location"] = map_location
        captured["strict"] = strict
        return "loaded-model"

    monkeypatch.setattr(RadarLightningModel, "load_from_checkpoint", classmethod(fake_loader))

    loaded = RadarLightningModel.from_checkpoint("/tmp/model.ckpt", device="cpu")

    assert loaded == "loaded-model"
    assert captured["checkpoint_path"] == "/tmp/model.ckpt"
    assert isinstance(captured["map_location"], torch.device)
    assert captured["map_location"].type == "cpu"
    assert captured["strict"] is True
