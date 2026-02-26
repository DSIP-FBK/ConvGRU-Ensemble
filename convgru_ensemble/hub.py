"""HuggingFace Hub integration for uploading and downloading ConvGRU-Ensemble models."""

import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def push_to_hub(
    checkpoint_path: str,
    repo_id: str,
    model_card_path: str | None = None,
    private: bool = False,
) -> str:
    """
    Upload a trained model checkpoint to HuggingFace Hub.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.ckpt`` checkpoint file.
    repo_id : str
        HuggingFace Hub repository ID (e.g., ``'it4lia/irene'``).
    model_card_path : str or None, optional
        Path to a model card markdown file. If provided, it is uploaded
        as ``README.md``. Default is ``None``.
    private : bool, optional
        Whether to create a private repository. Default is ``False``.

    Returns
    -------
    url : str
        URL of the uploaded model on HuggingFace Hub.
    """
    import torch

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Copy checkpoint
        shutil.copy2(checkpoint_path, tmp_path / "model.ckpt")

        # Extract and save model config from checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "hyper_parameters" in ckpt:
            hparams = ckpt["hyper_parameters"]
            # Convert non-serializable values to strings
            config = {}
            for k, v in hparams.items():
                try:
                    json.dumps(v)
                    config[k] = v
                except (TypeError, ValueError):
                    config[k] = str(v)
            with open(tmp_path / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        # Copy model card as README.md
        if model_card_path is not None:
            shutil.copy2(model_card_path, tmp_path / "README.md")

        url = api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            commit_message="Upload ConvGRU-Ensemble model",
        )

    return url


def from_pretrained(
    repo_id: str,
    filename: str = "model.ckpt",
    device: str = "cpu",
) -> "RadarLightningModel":  # noqa: F821
    """
    Download and load a pretrained model from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace Hub repository ID (e.g., ``'it4lia/irene'``).
    filename : str, optional
        Name of the checkpoint file in the repository. Default is
        ``'model.ckpt'``.
    device : str, optional
        Device to map the model weights to. Default is ``'cpu'``.

    Returns
    -------
    model : RadarLightningModel
        Model with loaded pretrained weights.
    """
    from .lightning_model import RadarLightningModel

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return RadarLightningModel.from_checkpoint(ckpt_path, device=device)
