from typing import Any, Dict, Optional

import torch
import torchvision
from torch import nn
import pytorch_lightning as pl

from model import EncoderDecoder
from losses import build_loss


def apply_radar_colormap(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert grayscale radar values (0-1, representing 0-60 dBZ) to RGB using STEPS-BE colorscale.

    Args:
        tensor: Grayscale tensor with values in [0, 1], shape (N, 1, H, W)

    Returns:
        RGB tensor with shape (N, 3, H, W)
    """
    # STEPS-BE colors (RGB values normalized to 0-1)
    colors = torch.tensor([
        [0, 255, 255],      # cyan
        [0, 191, 255],      # deepskyblue
        [30, 144, 255],     # dodgerblue
        [0, 0, 255],        # blue
        [127, 255, 0],      # chartreuse
        [50, 205, 50],      # limegreen
        [0, 128, 0],        # green
        [0, 100, 0],        # darkgreen
        [255, 255, 0],      # yellow
        [255, 215, 0],      # gold
        [255, 165, 0],      # orange
        [255, 0, 0],        # red
        [255, 0, 255],      # magenta
        [139, 0, 139],      # darkmagenta
    ], dtype=torch.float32, device=tensor.device) / 255.0

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
    Processes 
    
    Args
    ----
    input_channels : int
        Number input channels per grid point (ensemble members).
    forecast_steps: int
        Number of future steps to forecast.
    num_block : int
        Number of bloks in the Encoder and Decoder.
    lr : float
        Initial learning rate.
    """
    def __init__(
        self,
        input_channels: int,
        forecast_steps: int,
        num_blocks: int,
        ensemble_size: int = 1,
        noisy_decoder: bool = False,
        loss_class: Optional[type | str] = None,
        loss_params: Optional[Dict[str, Any]] = None,
        masked_loss: bool = False,
        optimizer_class: Optional[type] = None,
        optimizer_params: Optional[Dict[str, Any]] = None,
        lr_scheduler_class: Optional[type] = None,
        lr_scheduler_params: Optional[Dict[str, Any]] = None,
    ) -> None:
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

    def forward(self, x: torch.Tensor, ensemble_size: int | None = None) -> torch.Tensor:
        ensemble_size = self.hparams.ensemble_size if ensemble_size is None else ensemble_size
        return self.model(x, steps=self.hparams.forecast_steps, noisy_decoder=self.hparams.noisy_decoder, ensemble_size=ensemble_size)

    def shared_step(self, batch: Dict[str, torch.Tensor], split: str = "train", ensemble_size: int | None = None) -> torch.Tensor:
        data = batch['data']
        past = data[:, :-self.hparams.forecast_steps]
        future = data[:, -self.hparams.forecast_steps:]
        
        preds = self(past, ensemble_size=ensemble_size).clamp(min=-1, max=1)  # Ensure predictions are within [-1, 1]

        if self.hparams.masked_loss:
            mask = batch['mask'][:, -self.hparams.forecast_steps:]
            loss = self.criterion(preds, future, mask)
        else:
            loss = self.criterion(preds, future)
        
        # Handle tuple return from composite losses
        if isinstance(loss, tuple):
            loss, log_dict = loss
            # log_dict already contains split-prefixed keys like 'val/pixel_loss'
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=(split=="train"), on_epoch=True, sync_dist=True)

        self.log(f"{split}_loss", loss, prog_bar=True, on_epoch=True, on_step=(split=="train"), sync_dist=True)

        # Log ensemble diversity for ensemble training
        if self.hparams.ensemble_size > 1:
            ensemble_std = preds.std(dim=2).mean()  # std across ensemble members
            self.log(f"{split}_ensemble_std", ensemble_std, on_epoch=True, sync_dist=True)

        if split=="train" and (self.global_step in self.log_images_iterations or self.global_step % self.log_images_iterations[-1] == 0):
            self.log_images(past, future, preds, split=split)
        return loss
    
    def log_images(self, past: torch.Tensor, future: torch.Tensor, preds: torch.Tensor, split: str = "val") -> None:
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
        preds_sample = preds[sample_idx]    # (T, E, H, W) or (T, C, H, W)

        if self.hparams.ensemble_size > 1:
            # Layout: rows = [future, avg, member0, member1, ...], cols = timesteps
            preds_avg = preds_sample.mean(dim=1, keepdim=True)  # (T, E, H, W) -> (T, 1, H, W)
            num_members_to_log = min(3, preds_sample.shape[1])

            # Collect all rows: future first, then average, then individual members
            rows = [future_sample]  # (T, 1, H, W)
            rows.append(preds_avg)  # (T, 1, H, W)
            for i in range(num_members_to_log):
                rows.append(preds_sample[:, i:i+1, :, :])  # (T, 1, H, W)

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

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, split="train")
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int,) -> torch.Tensor:
        # fix ensemble size to 10 during validation for better estimate
        loss = self.shared_step(batch, split="val", ensemble_size=10)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, split="test", ensemble_size=10)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.hparams.optimizer_class is not None:
            optimizer = self.hparams.optimizer_class(self.parameters(), **self.hparams.optimizer_params) \
                if self.hparams.optimizer_params is not None \
                else self.hparams.optimizer_class(self.parameters())
            print(f"Using optimizer: {self.hparams.optimizer_class.__name__} with params {self.hparams.optimizer_params}")
        else:
            optimizer = torch.optim.Adam(self.parameters())
            print("Using default Adam optimizer with default parameters.")

        if self.hparams.lr_scheduler_class is not None:
            lr_scheduler = self.hparams.lr_scheduler_class(optimizer, **self.hparams.lr_scheduler_params) \
                if self.hparams.lr_scheduler_params is not None \
                else self.hparams.lr_scheduler_class(optimizer)
            print(f"Using LR scheduler: {self.hparams.lr_scheduler_class.__name__} with params {self.hparams.lr_scheduler_params}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val_loss"
                }
            }
        else:
            return {"optimizer": optimizer}


class RadarGANLightningModel(RadarLightningModel):
    """
    Radar nowcasting model with GAN loss for sharper predictions.

    Inherits from RadarLightningModel and adds GAN training with manual optimization.
    Supports ensemble mode for CRPS-GAN training where:
    - Generator produces M ensemble members with different noise inputs
    - CRPS loss is computed over the full ensemble for diversity
    - A random member is sampled for the discriminator for sharpness

    Args
    ----
    input_channels : int
        Number of input channels.
    forecast_steps : int
        Number of future steps to forecast.
    num_blocks : int
        Number of blocks in the Encoder and Decoder.
    ensemble_size : int
        Number of ensemble members to generate. Default 1 (no ensemble).
    noisy_decoder : bool
        Whether to use noise in decoder input. Required for ensemble diversity.
    gan_loss : nn.Module
        GANLoss instance containing the discriminator.
    lr : float
        Learning rate for both optimizers.
    """
    def __init__(
        self,
        input_channels: int,
        forecast_steps: int,
        num_blocks: int,
        ensemble_size: int = 1,
        noisy_decoder: bool = False,
        gan_loss: nn.Module = None,
        lr: float = 1e-4,
    ) -> None:
        # Initialize parent with dummy loss (we use gan_loss instead)
        super().__init__(
            input_channels=input_channels,
            forecast_steps=forecast_steps,
            num_blocks=num_blocks,
            ensemble_size=ensemble_size,
            noisy_decoder=noisy_decoder,
            loss_class="mse",  # Placeholder, not used
        )
        # Override for manual optimization
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['gan_loss'])

        # GAN loss (contains discriminator) - replaces self.criterion
        self.gan_loss = gan_loss

    def get_last_layer(self) -> torch.Tensor:
        """Returns the last layer of the decoder for adaptive weight calculation."""
        return self.model.decoder.blocks[-1].convgru.cell.out_gate.weight

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        gen_opt, disc_opt = self.optimizers()

        data = batch['data']
        past = data[:, :-self.hparams.forecast_steps]
        future = data[:, -self.hparams.forecast_steps:]
        mask = batch['mask'][:, -self.hparams.forecast_steps:] if self.gan_loss.pixel_loss_masked else None

        # Forward pass
        preds = self(past).clamp(min=-1, max=1)

        # Train generator
        self.toggle_optimizer(gen_opt)
        gen_loss, gen_log = self.gan_loss(
            preds, future, optimizer_idx=0, global_step=self.global_step,
            mask=mask, last_layer=self.get_last_layer(), split="train"
        )
        self.log("train/gen_loss", gen_loss, prog_bar=True, on_step=True, on_epoch=True)
        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        gen_opt.step()
        self.untoggle_optimizer(gen_opt)

        # Train discriminator
        self.toggle_optimizer(disc_opt)
        disc_loss, disc_log = self.gan_loss(
            preds.detach(), future, optimizer_idx=1,
            global_step=self.global_step, last_layer=self.get_last_layer(), split="train"
        )
        self.log("train/disc_loss", disc_loss, prog_bar=True, on_step=True, on_epoch=True)
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        disc_opt.step()
        self.untoggle_optimizer(disc_opt)

        # Log metrics
        log_dict = {**gen_log, **disc_log}
        log_dict = {k: v.to(past.device) if isinstance(v, torch.Tensor) else v for k, v in log_dict.items()}
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # Log images at specified intervals
        if self.global_step in self.log_images_iterations or self.global_step % self.log_images_iterations[-1] == 0:
            self.log_images(past, future, preds, split="train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        data = batch['data']
        past = data[:, :-self.hparams.forecast_steps]
        future = data[:, -self.hparams.forecast_steps:]
        mask = batch['mask'][:, -self.hparams.forecast_steps:] if self.gan_loss.pixel_loss_masked else None

        # Use fixed ensemble size for validation
        val_ensemble_size = 10 if self.hparams.ensemble_size > 1 else 1
        preds = self(past, ensemble_size=val_ensemble_size).clamp(min=-1, max=1)

        # Compute losses (no backprop)
        gen_loss, gen_log = self.gan_loss(
            preds, future, optimizer_idx=0, global_step=self.global_step,
            mask=mask, last_layer=self.get_last_layer(), split="val"
        )
        disc_loss, disc_log = self.gan_loss(
            preds, future, optimizer_idx=1,
            global_step=self.global_step, last_layer=self.get_last_layer(), split="val"
        )

        self.log("val/gen_loss", gen_loss, prog_bar=True, on_epoch=True)
        self.log("val/disc_loss", disc_loss, prog_bar=False, on_epoch=True)

        log_dict = {**gen_log, **disc_log}
        log_dict = {k: v.to(past.device) if isinstance(v, torch.Tensor) else v for k, v in log_dict.items()}
        self.log_dict(log_dict, prog_bar=False, logger=True, on_epoch=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data = batch['data']
        past = data[:, :-self.hparams.forecast_steps]
        future = data[:, -self.hparams.forecast_steps:]

        # Use fixed ensemble size for test
        test_ensemble_size = 10 if self.hparams.ensemble_size > 1 else 1
        preds = self(past, ensemble_size=test_ensemble_size).clamp(min=-1, max=1)

        # Compute MSE on ensemble mean if ensemble mode
        if test_ensemble_size > 1:
            preds_mean = preds.mean(dim=2, keepdim=True)
            mse_loss = nn.functional.mse_loss(preds_mean, future)
            ensemble_std = preds.std(dim=2).mean()
            self.log("test/ensemble_std", ensemble_std)
        else:
            mse_loss = nn.functional.mse_loss(preds, future)

        self.log("test/mse_loss", mse_loss)
        return mse_loss

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[Any]]:
        lr = self.hparams.lr

        # Generator optimizer
        gen_opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.9), foreach=True)

        # Discriminator optimizer
        disc_opt = torch.optim.Adam(self.gan_loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9), foreach=True)

        return [gen_opt, disc_opt], []

