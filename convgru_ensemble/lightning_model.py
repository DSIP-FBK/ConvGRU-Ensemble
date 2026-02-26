from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

from .losses import build_loss
from .model import EncoderDecoder
from .utils import normalized_to_rainrate, rainrate_to_normalized


def apply_radar_colormap(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert grayscale radar values to RGB using the STEPS-BE colorscale.

    Maps normalized values in [0, 1] (representing 0-60 dBZ) to a 14-color
    discrete colormap. Pixels below 10 dBZ are rendered as white.

    Parameters
    ----------
    tensor : torch.Tensor
        Grayscale tensor with values in [0, 1], of shape ``(N, 1, H, W)``.

    Returns
    -------
    rgb : torch.Tensor
        RGB tensor of shape ``(N, 3, H, W)`` with values in [0, 1].
    """
    # STEPS-BE colors (RGB values normalized to 0-1)
    colors = (
        torch.tensor(
            [
                [0, 255, 255],  # cyan
                [0, 191, 255],  # deepskyblue
                [30, 144, 255],  # dodgerblue
                [0, 0, 255],  # blue
                [127, 255, 0],  # chartreuse
                [50, 205, 50],  # limegreen
                [0, 128, 0],  # green
                [0, 100, 0],  # darkgreen
                [255, 255, 0],  # yellow
                [255, 215, 0],  # gold
                [255, 165, 0],  # orange
                [255, 0, 0],  # red
                [255, 0, 255],  # magenta
                [139, 0, 139],  # darkmagenta
            ],
            dtype=torch.float32,
            device=tensor.device,
        )
        / 255.0
    )

    # dBZ levels: 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 (11 levels, 10 intervals)
    # But we have 14 colors, so extend to cover 10-80 dBZ range with 5 dBZ steps
    # Normalized thresholds (0-1 maps to 0-60 dBZ)
    # We'll use 14 intervals from 10 dBZ onwards
    num_colors = len(colors)
    min_dbz_norm = 10 / 60  # ~0.167, below this is background
    max_dbz_norm = 1.0
    thresholds = torch.linspace(min_dbz_norm, max_dbz_norm, num_colors + 1, device=tensor.device)

    # Output tensor (N, 3, H, W) - initialize with white for values below 10 dBZ
    N, _, H, W = tensor.shape
    output = torch.ones(N, 3, H, W, dtype=torch.float32, device=tensor.device)

    # Apply colormap: find which bin each pixel falls into
    for i in range(num_colors - 1):
        mask = (tensor[:, 0] >= thresholds[i]) & (tensor[:, 0] < thresholds[i + 1])
        for c in range(3):
            output[:, c][mask] = colors[i, c]

    # Last color handles all values >= second-to-last threshold (inclusive of max)
    mask = tensor[:, 0] >= thresholds[num_colors - 1]
    for c in range(3):
        output[:, c][mask] = colors[-1, c]

    return output


class RadarLightningModel(pl.LightningModule):
    """
    PyTorch Lightning module for radar precipitation nowcasting.

    Wraps an :class:`EncoderDecoder` model and handles training, validation,
    and test steps including loss computation, ensemble generation, and
    TensorBoard image logging.

    Parameters
    ----------
    input_channels : int
        Number of input channels per grid point.
    num_blocks : int
        Number of encoder/decoder blocks in the model.
    ensemble_size : int, optional
        Number of ensemble members to generate. Default is ``1``.
    noisy_decoder : bool, optional
        Whether to use random noise as decoder input. Default is ``False``.
    forecast_steps : int or None, optional
        Number of future timesteps to forecast. Default is ``None``.
    loss_class : type, str, or None, optional
        Loss function class or its string name (see ``PIXEL_LOSSES``).
        Default is ``None`` (MSELoss).
    loss_params : dict or None, optional
        Keyword arguments for the loss constructor. Default is ``None``.
    masked_loss : bool, optional
        Whether to wrap the loss with :class:`MaskedLoss`. Default is
        ``False``.
    optimizer_class : type or None, optional
        Optimizer class. Default is ``None`` (Adam).
    optimizer_params : dict or None, optional
        Keyword arguments for the optimizer. Default is ``None``.
    lr_scheduler_class : type or None, optional
        Learning rate scheduler class. Default is ``None``.
    lr_scheduler_params : dict or None, optional
        Keyword arguments for the LR scheduler. Default is ``None``.
    """

    def __init__(
        self,
        input_channels: int,
        num_blocks: int,
        ensemble_size: int = 1,
        noisy_decoder: bool = False,
        forecast_steps: type | int | None = None,
        loss_class: type | str | None = None,
        loss_params: dict[str, Any] | None = None,
        masked_loss: bool = False,
        optimizer_class: type | None = None,
        optimizer_params: dict[str, Any] | None = None,
        lr_scheduler_class: type | None = None,
        lr_scheduler_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize RadarLightningModel.

        Parameters
        ----------
        input_channels : int
            Number of input channels per grid point.
        num_blocks : int
            Number of encoder/decoder blocks.
        ensemble_size : int, optional
            Number of ensemble members. Default is ``1``.
        noisy_decoder : bool, optional
            Use random noise as decoder input. Default is ``False``.
        forecast_steps : int or None, optional
            Number of future timesteps to forecast. Default is ``None``.
        loss_class : type, str, or None, optional
            Loss function class or name. Default is ``None``.
        loss_params : dict or None, optional
            Loss constructor kwargs. Default is ``None``.
        masked_loss : bool, optional
            Wrap loss with masking. Default is ``False``.
        optimizer_class : type or None, optional
            Optimizer class. Default is ``None``.
        optimizer_params : dict or None, optional
            Optimizer kwargs. Default is ``None``.
        lr_scheduler_class : type or None, optional
            LR scheduler class. Default is ``None``.
        lr_scheduler_params : dict or None, optional
            LR scheduler kwargs. Default is ``None``.
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.model = EncoderDecoder(self.hparams.input_channels, self.hparams.num_blocks)

        self.criterion = build_loss(
            loss_class=self.hparams.loss_class,
            loss_params=self.hparams.loss_params,
            masked_loss=self.hparams.masked_loss,
        )
        self.log_images_iterations = [50, 100, 200, 500, 750, 1000, 2000, 5000]

        if self.hparams.ensemble_size > 1:
            print(f"Using ensemble mode: {self.hparams.ensemble_size} independent ensemble members will be generated.")

    def forward(self, x: torch.Tensor, forecast_steps: int, ensemble_size: int | None = None) -> torch.Tensor:
        """
        Run the encoder-decoder forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, C, H, W)``.
        forecast_steps : int
            Number of future timesteps to forecast.
        ensemble_size : int or None, optional
            Number of ensemble members. If ``None``, uses the value from
            ``hparams``. Default is ``None``.

        Returns
        -------
        preds : torch.Tensor
            Predictions of shape ``(B, forecast_steps, C, H, W)`` or
            ``(B, forecast_steps, ensemble_size, H, W)`` for ensembles.
        """
        ensemble_size = self.hparams.ensemble_size if ensemble_size is None else ensemble_size
        return self.model(
            x, steps=forecast_steps, noisy_decoder=self.hparams.noisy_decoder, ensemble_size=ensemble_size
        )

    def shared_step(
        self, batch: dict[str, torch.Tensor], split: str = "train", ensemble_size: int | None = None
    ) -> torch.Tensor:
        """
        Shared forward step used during training, validation, and testing.

        Splits the input into past and future, runs the model, computes the
        loss, and logs metrics and optional images.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Batch dictionary with key ``'data'`` of shape
            ``(B, T_total, C, H, W)`` and optionally ``'mask'``.
        split : str, optional
            One of ``'train'``, ``'val'``, or ``'test'``. Controls logging
            behavior. Default is ``'train'``.
        ensemble_size : int or None, optional
            Override for the number of ensemble members. Default is ``None``.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value.
        """
        data = batch["data"]
        past = data[:, : -self.hparams.forecast_steps]
        future = data[:, -self.hparams.forecast_steps :]

        preds = self(past, forecast_steps=self.hparams.forecast_steps, ensemble_size=ensemble_size).clamp(
            min=-1, max=1
        )  # Ensure predictions are within [-1, 1]

        if self.hparams.masked_loss:
            mask = batch["mask"][:, -self.hparams.forecast_steps :]
            loss = self.criterion(preds, future, mask)
        else:
            loss = self.criterion(preds, future)

        # Handle tuple return from composite losses
        if isinstance(loss, tuple):
            loss, log_dict = loss
            # log_dict already contains split-prefixed keys like 'val/pixel_loss'
            self.log_dict(
                log_dict, prog_bar=False, logger=True, on_step=(split == "train"), on_epoch=True, sync_dist=True
            )

        self.log(f"{split}_loss", loss, prog_bar=True, on_epoch=True, on_step=(split == "train"), sync_dist=True)

        # Log ensemble diversity for ensemble training
        if self.hparams.ensemble_size > 1:
            ensemble_std = preds.std(dim=2).mean()  # std across ensemble members
            self.log(f"{split}_ensemble_std", ensemble_std, on_epoch=True, sync_dist=True)

        if split == "train" and (
            self.global_step in self.log_images_iterations or self.global_step % self.log_images_iterations[-1] == 0
        ):
            self.log_images(past, future, preds, split=split)
        return loss

    def log_images(self, past: torch.Tensor, future: torch.Tensor, preds: torch.Tensor, split: str = "val") -> None:
        """
        Log radar image grids to TensorBoard.

        Visualizes the first sample in the batch, showing past frames, ground
        truth future, ensemble average, and individual ensemble members using
        the STEPS-BE radar colormap.

        Parameters
        ----------
        past : torch.Tensor
            Past input frames of shape ``(B, T_past, C, H, W)``.
        future : torch.Tensor
            Ground truth future frames of shape ``(B, T_future, C, H, W)``.
        preds : torch.Tensor
            Predicted frames of shape ``(B, T_future, C_or_E, H, W)``.
        split : str, optional
            Split name used as TensorBoard tag prefix. Default is ``'val'``.
        """
        # Log first sample in the batch
        sample_idx = 0

        # Log past separately
        past_sample = past[sample_idx]
        if self.hparams.ensemble_size > 1:
            past_sample = past_sample.mean(dim=1, keepdim=True)
        past_norm = (past_sample + 1) / 2
        past_rgb = apply_radar_colormap(past_norm)
        past_grid = torchvision.utils.make_grid(past_rgb, nrow=past_sample.shape[0])
        self.logger.experiment.add_image(f"{split}/past", past_grid, self.global_step)

        # Create combined preds grid: future (ground truth) as first row, then avg + ensemble members
        future_sample = future[sample_idx]  # (T, C, H, W)
        preds_sample = preds[sample_idx]  # (T, E, H, W) or (T, C, H, W)

        if self.hparams.ensemble_size > 1:
            # Layout: rows = [future, avg, member0, member1, ...], cols = timesteps
            preds_avg = preds_sample.mean(dim=1, keepdim=True)  # (T, E, H, W) -> (T, 1, H, W)
            num_members_to_log = min(3, preds_sample.shape[1])

            # Collect all rows: future first, then average, then individual members
            rows = [future_sample]  # (T, 1, H, W)
            rows.append(preds_avg)  # (T, 1, H, W)
            for i in range(num_members_to_log):
                rows.append(preds_sample[:, i : i + 1, :, :])  # (T, 1, H, W)

            # Stack all rows: (num_rows * T, 1, H, W)
            all_frames = torch.cat(rows, dim=0)  # ((2 + num_members) * T, 1, H, W)
            all_frames_norm = (all_frames + 1) / 2
            all_frames_rgb = apply_radar_colormap(all_frames_norm)
            grid = torchvision.utils.make_grid(all_frames_rgb, nrow=future_sample.shape[0])
            self.logger.experiment.add_image(f"{split}/preds", grid, self.global_step)
        else:
            # For non-ensemble: show future and preds in two rows
            rows = [future_sample, preds_sample]  # Each is (T, C, H, W)
            all_frames = torch.cat(rows, dim=0)  # (2 * T, C, H, W)
            all_frames_norm = (all_frames + 1) / 2
            all_frames_rgb = apply_radar_colormap(all_frames_norm)
            grid = torchvision.utils.make_grid(all_frames_rgb, nrow=future_sample.shape[0])
            self.logger.experiment.add_image(f"{split}/preds", grid, self.global_step)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Execute a single training step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Training batch.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        loss : torch.Tensor
            Training loss.
        """
        loss = self.shared_step(batch, split="train")
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Execute a single validation step.

        Uses 10 ensemble members for evaluation.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Validation batch.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        loss : torch.Tensor
            Validation loss.
        """
        loss = self.shared_step(batch, split="val", ensemble_size=10)
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Execute a single test step.

        Uses 10 ensemble members for evaluation.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Test batch.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        loss : torch.Tensor
            Test loss.
        """
        loss = self.shared_step(batch, split="test", ensemble_size=10)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure the optimizer and optional learning rate scheduler.

        Falls back to Adam with default parameters if no optimizer is
        specified. If a scheduler is provided, it monitors ``val_loss``.

        Returns
        -------
        config : dict
            Dictionary with ``'optimizer'`` and optionally ``'lr_scheduler'``
            keys, as expected by PyTorch Lightning.
        """
        if self.hparams.optimizer_class is not None:
            optimizer = (
                self.hparams.optimizer_class(self.parameters(), **self.hparams.optimizer_params)
                if self.hparams.optimizer_params is not None
                else self.hparams.optimizer_class(self.parameters())
            )
            print(
                f"Using optimizer: {self.hparams.optimizer_class.__name__} with params {self.hparams.optimizer_params}"
            )
        else:
            optimizer = torch.optim.Adam(self.parameters())
            print("Using default Adam optimizer with default parameters.")

        if self.hparams.lr_scheduler_class is not None:
            lr_scheduler = (
                self.hparams.lr_scheduler_class(optimizer, **self.hparams.lr_scheduler_params)
                if self.hparams.lr_scheduler_params is not None
                else self.hparams.lr_scheduler_class(optimizer)
            )
            print(
                f"Using LR scheduler: {self.hparams.lr_scheduler_class.__name__} with params {self.hparams.lr_scheduler_params}"
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}
        else:
            return {"optimizer": optimizer}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "RadarLightningModel":
        """
        Load a model from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to the ``.ckpt`` checkpoint file.
        device : str, optional
            Device to map the checkpoint weights to. Default is ``'cpu'``.

        Returns
        -------
        model : RadarLightningModel
            Model with loaded weights.
        """
        return cls.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device(device),
            strict=True,
            weights_only=False,
        )

    @classmethod
    def from_pretrained(cls, repo_id: str, filename: str = "model.ckpt", device: str = "cpu") -> "RadarLightningModel":
        """
        Load a pretrained model from HuggingFace Hub.

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
        from .hub import from_pretrained

        return from_pretrained(repo_id, filename, device)

    def predict(self, past: torch.Tensor, forecast_steps: int = 1, ensemble_size: int | None = 1) -> torch.Tensor:
        """
        Generate precipitation forecasts from past radar observations.

        Handles padding, NaN removal, unit conversion, and reshaping
        automatically. Input should be raw rain rate values.

        Parameters
        ----------
        past : torch.Tensor
            Past radar frames as rain rate in mm/h, of shape ``(T, H, W)``.
        forecast_steps : int, optional
            Number of future timesteps to forecast. Default is ``1``.
        ensemble_size : int, optional
            Number of ensemble members to generate. If ``None``, uses the
            value from ``hparams``. Default is ``1``.

        Returns
        -------
        preds : np.ndarray
            Forecasted rain rate in mm/h, of shape
            ``(ensemble_size, forecast_steps, H, W)``.

        Raises
        ------
        ValueError
            If ``past`` does not have exactly 3 dimensions.
        """
        if len(past.shape) != 3:
            raise ValueError("Input must be of shape (T, H, W)")

        T, H, W = past.shape
        ensemble_size = self.hparams.ensemble_size if ensemble_size is None else ensemble_size

        # Each block the model decrease the resolution by a factor of 2
        # The input must be divisible by 2^(num_blocks-1)
        divisor = 2 ** (self.hparams.num_blocks)
        padH = (divisor - (H % divisor)) % divisor
        padW = (divisor - (W % divisor)) % divisor
        padded_past = past
        if padH != 0 or padW != 0:
            padded_past = np.pad(past, ((0, 0), (0, padH), (0, padW)), mode="constant", constant_values=0)

        # Remove Nan
        past_clean = np.nan_to_num(padded_past)

        # Reshape the input to (B, T, C, H, W)
        past_clean = past_clean[np.newaxis, :, np.newaxis, ...]

        # Rainrate to normalized reflectivity
        norm_past = rainrate_to_normalized(past_clean)

        # Numpy to torch tensor
        x = torch.from_numpy(norm_past)

        # Move to device
        x = x.to(self.device)

        # Forward pass
        self.eval()
        with torch.no_grad():
            preds = self.model(x, forecast_steps, self.hparams.noisy_decoder, ensemble_size)

        # Move to CPU
        preds = preds.cpu()

        # Tensor to numpy array
        preds = preds.numpy()

        # Rescale back to rain rate
        preds = normalized_to_rainrate(preds)

        # Remove the batch (T, E, H, W)
        preds = preds.squeeze(0)

        # Swap the Time and Ensemble dimensions (E, T, H, W)
        preds = np.swapaxes(preds, 0, 1)

        # Remove the padding
        preds = preds[..., :H, :W]

        return preds
