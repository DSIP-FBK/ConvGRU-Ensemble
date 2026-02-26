from unittest.mock import MagicMock, patch

from convgru_ensemble.hub import from_pretrained


@patch("convgru_ensemble.hub.hf_hub_download")
@patch("convgru_ensemble.hub.RadarLightningModel", create=True)
def test_from_pretrained_calls_hf_hub_download(mock_model_cls, mock_download):
    mock_download.return_value = "/tmp/cached/model.ckpt"

    # Patch the import inside the function
    with patch("convgru_ensemble.lightning_model.RadarLightningModel") as mock_cls:
        mock_cls.from_checkpoint.return_value = MagicMock()
        from_pretrained("it4lia/irene", filename="model.ckpt", device="cpu")

    mock_download.assert_called_once_with(repo_id="it4lia/irene", filename="model.ckpt")
