import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .utils import rainrate_to_normalized


class SampledRadarDataset(Dataset):
    """
    PyTorch dataset that loads radar datacubes from a Zarr store using
    pre-sampled spatial-temporal coordinates from a CSV file.

    Each sample is a spatio-temporal datacube of shape ``(T, 1, H, W)``
    converted from rain rate to normalized reflectivity.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset containing the ``'RR'`` rain rate variable.
    csv_path : str
        Path to the CSV file with columns ``(t, x, y)`` specifying the
        top-left corner of each datacube.
    steps : int
        Number of timesteps to extract per sample.
    return_mask : bool, optional
        If ``True``, also return a spatial NaN mask. Default is ``False``.
    deterministic : bool, optional
        If ``True``, use a fixed random seed (42) for reproducibility.
        Default is ``False``.
    augment : bool, optional
        If ``True``, apply random spatial augmentations (rotation, flips).
        Default is ``False``.
    indices : sequence of int or None, optional
        Subset of row indices to use from the CSV. If ``None``, use all rows.
        Default is ``None``.
    """

    def __init__(
        self,
        zarr_path: str,
        csv_path: str,
        steps: int,
        return_mask: bool = False,
        deterministic: bool = False,
        augment: bool = False,
        indices=None,
    ):
        """
        Initialize SampledRadarDataset.

        Parameters
        ----------
        zarr_path : str
            Path to the Zarr dataset containing the ``'RR'`` rain rate
            variable.
        csv_path : str
            Path to the CSV file with columns ``(t, x, y)``.
        steps : int
            Number of timesteps to extract per sample.
        return_mask : bool, optional
            If ``True``, also return a spatial NaN mask. Default is ``False``.
        deterministic : bool, optional
            If ``True``, use a fixed random seed (42). Default is ``False``.
        augment : bool, optional
            If ``True``, apply random spatial augmentations. Default is
            ``False``.
        indices : sequence of int or None, optional
            Subset of row indices from the CSV. Default is ``None``.
        """
        self.coords = pd.read_csv(csv_path).sort_values("t")
        if indices is not None:
            self.coords = self.coords.iloc[list(indices)].reset_index(drop=True)
        self.zg = xr.open_zarr(zarr_path)
        self.RR = self.zg["RR"]
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
        """
        Return the number of samples in the dataset.

        Returns
        -------
        length : int
            Number of datacube samples.
        """
        return len(self.coords)

    def shape(self):
        """
        Return the nominal shape of the full dataset.

        Returns
        -------
        shape : tuple of int
            ``(num_samples, steps, 1, width, height)``.
        """
        return (len(self.coords), self.steps, 1, self.w, self.h)

    def _apply_augmentations(
        self, *tensors, rotate_prob: float = 0.5, hflip_prob: float = 0.5, vflip_prob: float = 0.5
    ):
        """
        Apply random spatial augmentations consistently to all input tensors.

        All tensors receive the same random transformation so that spatial
        alignment is preserved (e.g. between data and mask).

        Parameters
        ----------
        *tensors : torch.Tensor
            One or more tensors of shape ``(T, C, H, W)``.
        rotate_prob : float, optional
            Probability of applying a random 90-degree rotation. Default is
            ``0.5``.
        hflip_prob : float, optional
            Probability of applying a horizontal flip. Default is ``0.5``.
        vflip_prob : float, optional
            Probability of applying a vertical flip. Default is ``0.5``.

        Returns
        -------
        augmented : torch.Tensor or tuple of torch.Tensor
            Single tensor if one input was given, otherwise a tuple of
            augmented tensors.
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
        """
        Load and return a single datacube sample.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.

        Returns
        -------
        sample : dict of str to torch.Tensor
            Dictionary with key ``'data'`` containing a tensor of shape
            ``(T, 1, H, W)``. If ``return_mask`` is ``True``, also contains
            ``'mask'`` of shape ``(1, 1, H, W)``.
        """
        t0, x0, y0 = self.coords.iloc[idx]

        x_slice = slice(x0, x0 + self.w)
        y_slice = slice(y0, y0 + self.h)

        if self.steps < self.dt:
            # radom sampling within available time window
            t_start = self.rng.integers(t0, t0 + self.dt - self.steps + 1)
        else:
            t_start = t0
        t_slice = slice(t_start, t_start + self.steps)

        data = rainrate_to_normalized(self.RR[t_slice, x_slice, y_slice])

        # create a mask for all nan values over time dimension
        # shape: (1, H, W) - NOT repeated over time, broadcasting handles it
        if self.return_mask:
            mask = (~(np.isnan(data).any(axis=0, keepdims=True))).astype(np.float32)

        # replace nan values with -1
        data = np.nan_to_num(data, nan=-1.0)

        # convert to tensors
        data = torch.from_numpy(data[:, np.newaxis, :, :])
        if self.return_mask:
            mask = torch.from_numpy(mask.values[:, np.newaxis, :, :])

        # apply augmentations (training only)
        if self.augment:
            if self.return_mask:
                data, mask = self._apply_augmentations(data, mask)
            else:
                data = self._apply_augmentations(data)

        if self.return_mask:
            return {"data": data, "mask": mask}
        else:
            return {"data": data}


class RadarDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for radar datacube datasets.

    Handles train/val/test splitting and DataLoader creation from a single
    Zarr store and CSV coordinate file.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    csv_path : str
        Path to the CSV file with datacube coordinates.
    steps : int
        Number of timesteps per sample.
    train_ratio : float, optional
        Fraction of data used for training. Default is ``0.7``.
    val_ratio : float, optional
        Fraction of data used for validation. Default is ``0.15``.
    return_mask : bool, optional
        Whether to return NaN masks. Default is ``False``.
    deterministic : bool, optional
        Whether to use fixed random seeds. Default is ``False``.
    augment : bool, optional
        Whether to apply data augmentation (training set only). Default is
        ``True``.
    **dataloader_kwargs
        Additional keyword arguments forwarded to ``DataLoader`` (e.g.
        ``batch_size``, ``num_workers``, ``pin_memory``).
    """

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
        """
        Initialize RadarDataModule.

        Parameters
        ----------
        zarr_path : str
            Path to the Zarr dataset.
        csv_path : str
            Path to the CSV file with datacube coordinates.
        steps : int
            Number of timesteps per sample.
        train_ratio : float, optional
            Fraction of data for training. Default is ``0.7``.
        val_ratio : float, optional
            Fraction of data for validation. Default is ``0.15``.
        return_mask : bool, optional
            Whether to return NaN masks. Default is ``False``.
        deterministic : bool, optional
            Whether to use fixed random seeds. Default is ``False``.
        augment : bool, optional
            Whether to apply data augmentation. Default is ``True``.
        **dataloader_kwargs
            Forwarded to ``DataLoader``.
        """
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
        """
        Create train, validation, and test datasets from the CSV coordinates.

        Splits are chronological: the first ``train_ratio`` fraction is used
        for training, the next ``val_ratio`` for validation, and the rest for
        testing. Augmentation is only applied to the training set.

        Parameters
        ----------
        stage : str or None, optional
            Lightning stage (``'fit'``, ``'test'``, etc.). Ignored; all
            datasets are always created. Default is ``None``.
        """
        # Load CSV to get total length for splitting
        coords = pd.read_csv(self.csv_path).sort_values("t")
        n = len(coords)

        # Compute split indices
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # Create separate datasets (augmentation only for training)
        self.train_dataset = SampledRadarDataset(
            self.zarr_path,
            self.csv_path,
            self.steps,
            self.return_mask,
            self.deterministic,
            augment=self.augment,
            indices=range(0, train_end),
        )
        self.val_dataset = SampledRadarDataset(
            self.zarr_path,
            self.csv_path,
            self.steps,
            self.return_mask,
            self.deterministic,
            augment=False,
            indices=range(train_end, val_end),
        )
        self.test_dataset = SampledRadarDataset(
            self.zarr_path,
            self.csv_path,
            self.steps,
            self.return_mask,
            self.deterministic,
            augment=False,
            indices=range(val_end, n),
        )

    def train_dataloader(self):
        """
        Return the training DataLoader.

        Returns
        -------
        loader : DataLoader
            DataLoader over the training dataset with shuffling enabled.
        """
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        """
        Return the validation DataLoader.

        Returns
        -------
        loader : DataLoader
            DataLoader over the validation dataset without shuffling.
        """
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self):
        """
        Return the test DataLoader.

        Returns
        -------
        loader : DataLoader
            DataLoader over the test dataset without shuffling.
        """
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_kwargs)
