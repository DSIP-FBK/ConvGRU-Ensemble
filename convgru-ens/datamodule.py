import numpy as np
import pandas as pd
import zarr
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import time

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


class SampledRadarDataset(Dataset):
    def __init__(self, zarr_path: str, csv_path: str, steps: int, return_mask: bool = False, deterministic: bool = False, augment: bool = False, indices=None):
        self.coords = pd.read_csv(csv_path).sort_values('t')
        if indices is not None:
            self.coords = self.coords.iloc[list(indices)].reset_index(drop=True)
        self.zg = zarr.open(zarr_path, mode='r')
        self.RR = self.zg['RR']
        self.rng = np.random.default_rng(seed=42) if deterministic else np.random.default_rng(int(time.time()))
        self.return_mask = return_mask
        self.augment = augment

        if augment:
            print("Data augmentation is enabled.")

        # default valid grid size and time step
        self.w = 256
        self.h = 256
        self.dt = 24
        self.steps = steps

        # raise warning if steps > dt
        if self.steps > self.dt:
            print(f"Warning: requested steps ({self.steps}) > sampled time window ({self.dt})")
    
    def __len__(self):
        return len(self.coords)
    
    def shape(self):
        return (len(self.coords), self.steps, 1, self.w, self.h)

    def _apply_augmentations(self, *tensors,
                             rotate_prob: float = 0.5,
                             hflip_prob: float = 0.5,
                             vflip_prob: float = 0.5):
        """Apply random augmentations consistently across all timesteps to all input tensors.

        Args:
            *tensors: Tensors of shape (T, C, H, W)
        Returns:
            Single tensor if one input, otherwise tuple of augmented tensors
        """
        # Random 90-degree rotation (0, 90, 180, or 270 degrees)
        if self.rng.random() < rotate_prob:
            k = self.rng.integers(1, 4)  # 1=90, 2=180, 3=270 degrees
            tensors = [torch.rot90(t, k, dims=[-2, -1]) for t in tensors]

        # Random horizontal flip
        if self.rng.random() < hflip_prob:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]

        # Random vertical flip
        if self.rng.random() < vflip_prob:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]

        tensors = [t.contiguous() for t in tensors]
        return tensors[0] if len(tensors) == 1 else tuple(tensors)

    def __getitem__(self, idx: int):
        t0, x0, y0 = self.coords.iloc[idx]
        
        x_slice = slice(x0, x0 + self.w)
        y_slice = slice(y0, y0 + self.h)

        if self.steps < self.dt:
            # radom sampling within available time window
            t_start = self.rng.integers(t0, t0 + self.dt - self.steps + 1)
        else:
            t_start = t0
        t_slice = slice(t_start, t_start + self.steps)
        
        data = normalize_reflectivity(rainrate_to_reflectivity(self.RR[t_slice, x_slice, y_slice]))

        # create a mask for all nan values over time dimension
        # shape: (1, H, W) - NOT repeated over time, broadcasting handles it
        if self.return_mask:
            mask = (~(np.isnan(data).any(axis=0, keepdims=True))).astype(np.float32)

        # replace nan values with -1
        data = np.nan_to_num(data, nan=-1.0)

        # convert to tensors
        data = torch.from_numpy(data[:, np.newaxis, :, :])
        if self.return_mask:
            mask = torch.from_numpy(mask[:, np.newaxis, :, :])

        # apply augmentations (training only)
        if self.augment:
            if self.return_mask:
                data, mask = self._apply_augmentations(data, mask)
            else:
                data = self._apply_augmentations(data)

        if self.return_mask:
            return {'data': data, 'mask': mask}
        else:
            return {'data': data}
        
        
class RadarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        zarr_path,
        csv_path,
        steps,
        train_ratio=0.7,
        val_ratio=0.15,
        return_mask=False,
        deterministic=False,
        augment=True,
        **dataloader_kwargs,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.csv_path = csv_path
        self.steps = steps
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.dataloader_kwargs = dataloader_kwargs
        self.return_mask = return_mask
        self.deterministic = deterministic
        self.augment = augment

    def setup(self, stage=None):
        # Load CSV to get total length for splitting
        coords = pd.read_csv(self.csv_path).sort_values('t')
        n = len(coords)

        # Compute split indices
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # Create separate datasets (augmentation only for training)
        self.train_dataset = SampledRadarDataset(
            self.zarr_path, self.csv_path, self.steps, self.return_mask, self.deterministic,
            augment=self.augment, indices=range(0, train_end)
        )
        self.val_dataset = SampledRadarDataset(
            self.zarr_path, self.csv_path, self.steps, self.return_mask, self.deterministic,
            augment=False, indices=range(train_end, val_end)
        )
        self.test_dataset = SampledRadarDataset(
            self.zarr_path, self.csv_path, self.steps, self.return_mask, self.deterministic,
            augment=False, indices=range(val_end, n)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True, 
            **self.dataloader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.dataloader_kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self.dataloader_kwargs
        )