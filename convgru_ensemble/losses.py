from typing import Any

import torch
from torch import nn


class LossWithReduction(nn.Module):
    """
    Base class for losses with reduction options.

    Parameters
    ----------
    reduction : str, optional
        Reduction mode to apply to the loss. Must be one of ``'mean'``,
        ``'sum'``, or ``'none'``. Default is ``'mean'``.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize LossWithReduction.

        Parameters
        ----------
        reduction : str, optional
            Reduction mode to apply to the loss. Must be one of ``'mean'``,
            ``'sum'``, or ``'none'``. Default is ``'mean'``.
        """
        super().__init__()
        assert reduction in ["mean", "sum", "none"], "reduction must be 'mean', 'sum', or 'none'"
        self.reduction = reduction

    def apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified reduction to the loss tensor.

        Parameters
        ----------
        loss : torch.Tensor
            Loss tensor to reduce.

        Returns
        -------
        reduced_loss : torch.Tensor
            Reduced loss tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class MaskedLoss(LossWithReduction):
    """
    Wrapper to apply a mask to a given loss function.

    Masks out invalid pixels before computing the loss, ensuring that only
    valid regions contribute to the final value.

    Parameters
    ----------
    elementwise_loss : nn.Module
        Base loss function to be masked. Must accept ``(preds, target)`` and
        return element-wise (unreduced) loss. Should be instantiated with
        ``reduction='none'``.
    reduction : str, optional
        Reduction mode applied after masking. Must be one of ``'mean'``,
        ``'sum'``, or ``'none'``. Default is ``'mean'``.
    """

    def __init__(self, elementwise_loss: nn.Module, reduction: str = "mean"):
        """
        Initialize MaskedLoss.

        Parameters
        ----------
        elementwise_loss : nn.Module
            Base loss function to be masked. Must accept ``(preds, target)``
            and return element-wise (unreduced) loss. Should be instantiated
            with ``reduction='none'``.
        reduction : str, optional
            Reduction mode applied after masking. Must be one of ``'mean'``,
            ``'sum'``, or ``'none'``. Default is ``'mean'``.
        """
        super().__init__(reduction=reduction)
        self.elementwise_loss = elementwise_loss

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked loss.

        Parameters
        ----------
        preds : torch.Tensor
            Predictions of shape (B, T, C, *D).
        target : torch.Tensor
            Target of shape (B, T, C, *D).
        mask : torch.Tensor
            Mask of shape (B, T, C, *D)
                       or (B, T, 1, *D)
                       or (B, 1, 1, *D), with 1 for valid and 0 for invalid pixels.
            Broadcasted to match preds/target shape if needed.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value.
        """
        # assert preds.shape == target.shape, f"preds and target must have the same shape, got {preds.shape} and {target.shape}"
        # assert mask.shape == preds.shape, f"mask must have the same shape as preds, got {mask.shape} and {preds.shape}"

        # Compute element-wise loss
        elementwise_loss = self.elementwise_loss(preds, target)  # shape (B, T, C, *D)

        # Apply mask (broadcast if needed)
        masked_loss = elementwise_loss * mask  # shape (B, T, C, *D)

        # Average over valid pixels
        # Account for broadcasting: mask.sum() × broadcast_factor
        broadcast_factor = elementwise_loss.numel() // mask.numel()
        valid_pixels = mask.sum() * broadcast_factor
        if valid_pixels > 0:
            if self.reduction == "mean":
                return masked_loss.sum() / valid_pixels
            elif self.reduction == "sum":
                return masked_loss.sum()
            else:  # 'none'
                return masked_loss
        else:
            return torch.tensor(0.0, device=preds.device)


class CRPS(LossWithReduction):
    r"""
    Continuous Ranked Probability Score (CRPS) loss with optional temporal
    consistency regularization.

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|], where X, X' are independent
    samples from the forecast distribution and y is the observation.

    Parameters
    ----------
    temporal_lambda : float, optional
        Weight for the temporal consistency penalty. If ``0.0`` (default),
        the penalty is disabled. When enabled, adds a penalty for large
        differences between consecutive timesteps within each ensemble
        member, preventing pulsing artifacts.
    reduction : str, optional
        Reduction mode. Must be one of ``'mean'``, ``'sum'``, or ``'none'``.
        ``'mean'`` averages over batch and all non-ensemble dimensions.
        Default is ``'mean'``.

    Expected shapes
    ---------------
    preds : (B, T, M, \*D)
        Ensemble predictions with time T on dim=1, ensemble size M on dim=2.
    target : (B, T, C, \*D)
        Deterministic target / analysis with channel C on dim=2 (should be 1).
    """

    def __init__(self, temporal_lambda: float = 0.0, reduction: str = "mean"):
        """
        Initialize CRPS loss.

        Parameters
        ----------
        temporal_lambda : float, optional
            Weight for the temporal consistency penalty. Default is ``0.0``
            (disabled).
        reduction : str, optional
            Reduction mode. Must be one of ``'mean'``, ``'sum'``, or
            ``'none'``. Default is ``'mean'``.
        """
        super().__init__(reduction=reduction)
        self.temporal_lambda = temporal_lambda

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CRPS loss.

        CRPS = E[|X - y|] - 0.5 * E[|X - X'|], where X, X' are independent
        samples from the forecast distribution and y is the observation.

        Parameters
        ----------
        preds : torch.Tensor
            Ensemble forecasts of shape ``(B, T, M, *D)``, where B is batch
            size, T is the number of timesteps, M is ensemble size, and
            ``*D`` are spatial dimensions.
        target : torch.Tensor
            Verifying observation / analysis of shape ``(B, T, C, *D)``,
            where C should be 1. Broadcasts against ``preds`` on dim=2.

        Returns
        -------
        loss : torch.Tensor
            Scalar if ``reduction='mean'`` or ``'sum'``, otherwise a tensor
            of shape ``(B, T, 1, *D)`` (or ``(B, 1, 1, *D)`` when
            ``temporal_lambda > 0``).
        """
        # preds: (B, T, M, *D)
        # target: (B, T, C, *D) where C should be 1
        # target broadcasts against preds: (B, T, 1, *D) vs (B, T, M, *D)

        # First term: E[|X - y|]
        # Compute absolute difference between each ensemble member and target
        diff_to_target = torch.abs(preds - target)  # (B, T, M, *D)
        term1 = diff_to_target.mean(dim=2)  # Average over ensemble: (B, T, *D)

        # Second term: 0.5 * E[|X - X'|]
        # Compute pairwise differences between ensemble members
        # preds: (B, T, M, *D)
        # Expand for pairwise differences
        # preds_i: (B, T, M, 1, *D)
        # preds_j: (B, T, 1, M, *D)
        preds_i = preds.unsqueeze(3)
        preds_j = preds.unsqueeze(2)

        # Pairwise absolute differences: (B, T, M, M, *D)
        pairwise_diff = torch.abs(preds_i - preds_j)

        # Average over both ensemble dimensions
        # Sum over M*M pairs and divide by M*M
        term2 = 0.5 * pairwise_diff.mean(dim=(2, 3))  # (B, T, *D)

        # CRPS
        crps = term1 - term2  # (B, T, *D)

        # Temporal consistency penalty
        if self.temporal_lambda > 0:
            # average over time dimension
            crps = crps.mean(dim=1)  # (B, *D)

            # preds: (B, T, M, *D)
            # Compute differences between consecutive timesteps per ensemble member
            temporal_diff = preds[:, 1:, :, ...] - preds[:, :-1, :, ...]  # (B, T-1, M, *D)
            temporal_penalty = torch.abs(temporal_diff).mean(
                dim=(1, 2)
            )  # average over time and ensemble dimensions (B, *D)
            # Add penalty to CRPS (before reduction, averaged over time)
            crps = crps + self.temporal_lambda * temporal_penalty
            crps = crps[:, None, None, ...]  # add time and channel dims back for consistency (B, 1, 1, *D)
        else:
            # Keep singleton channel dim for MaskedLoss compatibility: (B, T, 1, *D)
            crps = crps.unsqueeze(2)

        return self.apply_reduction(crps)


class afCRPS(LossWithReduction):
    r"""
    Almost fair CRPS (afCRPS) loss as in eq. (4) of Lang et al. (2024).

    Interpolates between the standard (energy-score style) CRPS and the
    fair CRPS via the fairness parameter ``alpha``.

    Parameters
    ----------
    alpha : float, optional
        Fairness parameter in ``(0, 1]``. ``alpha=1`` recovers the fair
        CRPS. Lang et al. (2024) recommend ``alpha=0.95``. Default is
        ``0.95``.
    temporal_lambda : float, optional
        Weight for the temporal consistency penalty. If ``0.0`` (default),
        the penalty is disabled. When enabled, adds a penalty for large
        differences between consecutive timesteps within each ensemble
        member.
    reduction : str, optional
        Reduction mode. Must be one of ``'mean'``, ``'sum'``, or ``'none'``.
        ``'mean'`` averages over batch and all non-ensemble dimensions.
        Default is ``'mean'``.

    Expected shapes
    ---------------
    preds : (B, T, M, \*D)
        Ensemble predictions with time T on dim=1, ensemble size M on dim=2.
    target : (B, T, C, \*D)
        Deterministic target / analysis with channel C on dim=2 (should be 1).
    """

    def __init__(self, alpha: float = 0.95, temporal_lambda: float = 0.0, reduction: str = "mean"):
        """
        Initialize afCRPS loss.

        Parameters
        ----------
        alpha : float, optional
            Fairness parameter in ``(0, 1]``. ``alpha=1`` recovers the fair
            CRPS. Default is ``0.95``.
        temporal_lambda : float, optional
            Weight for the temporal consistency penalty. Default is ``0.0``
            (disabled).
        reduction : str, optional
            Reduction mode. Must be one of ``'mean'``, ``'sum'``, or
            ``'none'``. Default is ``'mean'``.

        Raises
        ------
        ValueError
            If ``alpha`` is not in ``(0, 1]``.
        """
        super().__init__(reduction=reduction)
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self.temporal_lambda = temporal_lambda

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute afCRPS over an ensemble.

        Parameters
        ----------
        preds : torch.Tensor
            (B, T, M, *D) ensemble forecasts.
        target : torch.Tensor
            (B, T, C, *D) verifying observation / analysis (C=1).

        Returns
        -------
        loss : torch.Tensor
            Scalar if reduction=='mean', else per-sample tensor.
        """
        if preds.dim() < 3:
            raise ValueError("preds must have at least 3 dimensions.")
        if target.shape[0] != preds.shape[0]:
            raise ValueError("batch dimension of preds and target must match.")
        if target.shape[1] != preds.shape[1]:
            raise ValueError("time dimension of preds and target must match.")

        # preds: (B, T, M, *D), target: (B, T, 1, *D)
        M = preds.shape[2]
        if M < 2:
            raise ValueError("Ensemble size M must be >= 2 for afCRPS.")

        eps = (1.0 - self.alpha) / float(M)

        # |x_j - y| : (B, T, M, *D)  — target broadcasts via C=1
        abs_x_minus_y = (preds - target).abs()

        # Pairwise terms over ensemble dim (dim=2), j != k
        #   x_j: (B, T, M, 1, *D)
        #   x_k: (B, T, 1, M, *D)
        x_j = preds.unsqueeze(3)
        x_k = preds.unsqueeze(2)

        # |x_j - y|, |x_k - y| broadcast to (B, T, M, M, *D)
        abs_xj_minus_y = abs_x_minus_y.unsqueeze(3)
        abs_xk_minus_y = abs_x_minus_y.unsqueeze(2)

        # |x_j - x_k|: (B, T, M, M, *D)
        abs_xj_minus_xk = (x_j - x_k).abs()

        # Per (j,k) term: |x_j - y| + |x_k - y| - (1 - eps)|x_j - x_k|
        term = abs_xj_minus_y + abs_xk_minus_y - (1.0 - eps) * abs_xj_minus_xk

        # Exclude j == k (diagonal) since eq. (4) sums over k != j
        idx = torch.arange(M, device=preds.device)
        mask = idx[:, None] != idx[None, :]  # (M, M)
        term = term * mask.view(1, 1, M, M, *([1] * (term.dim() - 4)))

        # Sum over j and k dims → (B, T, *D)
        summed = term.sum(dim=(2, 3))

        # Normalization factor 1 / [2 M (M - 1)]
        afcrps = summed / (2.0 * M * (M - 1))  # (B, T, *D)

        # Temporal consistency penalty
        if self.temporal_lambda > 0:
            # average over time dimension
            afcrps = afcrps.mean(dim=1)  # (B, *D)

            # preds: (B, T, M, *D)
            # Compute differences between consecutive timesteps per ensemble member
            temporal_diff = preds[:, 1:, :, ...] - preds[:, :-1, :, ...]  # (B, T-1, M, *D)
            temporal_penalty = torch.abs(temporal_diff).mean(
                dim=(1, 2)
            )  # average over time and ensemble dimensions (B, *D)
            # Add penalty to afCRPS (before reduction, averaged over time)
            afcrps = afcrps + self.temporal_lambda * temporal_penalty
            afcrps = afcrps[:, None, None, ...]  # add time and channel dims back for consistency (B, 1, 1, *D)
        else:
            # Keep singleton channel dim for MaskedLoss compatibility: (B, T, 1, *D)
            afcrps = afcrps.unsqueeze(2)

        return self.apply_reduction(afcrps)


PIXEL_LOSSES = {"mse": nn.MSELoss, "mae": nn.L1Loss, "crps": CRPS, "afcrps": afCRPS}


def build_loss(
    loss_class: type | str,
    loss_params: dict[str, Any] | None = None,
    masked_loss: bool = False,
) -> nn.Module:
    """
    Build a loss function, optionally wrapped with masking.

    Resolves a loss class by name (from ``PIXEL_LOSSES``) or accepts a class
    directly, instantiates it with the given parameters, and optionally wraps
    it in a :class:`MaskedLoss`.

    Parameters
    ----------
    loss_class : type or str
        Loss class or its string name. Accepted string names are the keys of
        ``PIXEL_LOSSES``: ``'mse'``, ``'mae'``, ``'crps'``, ``'afcrps'``.
        If ``None``, defaults to ``nn.MSELoss``.
    loss_params : dict of str to any, optional
        Keyword arguments forwarded to the loss class constructor. If
        ``masked_loss`` is ``True``, the ``'reduction'`` key (if present)
        is extracted and passed to :class:`MaskedLoss` instead. Default is
        ``None``.
    masked_loss : bool, optional
        If ``True``, the loss is wrapped in :class:`MaskedLoss` and the
        inner loss is instantiated with ``reduction='none'``. Default is
        ``False``.

    Returns
    -------
    criterion : nn.Module
        Instantiated loss module, optionally wrapped in :class:`MaskedLoss`.

    Raises
    ------
    ValueError
        If ``loss_class`` is a string not found in ``PIXEL_LOSSES``.
    """
    if isinstance(loss_class, str):
        if loss_class.lower() not in PIXEL_LOSSES:
            raise ValueError(f"Unknown loss class '{loss_class}'. Available: {list(PIXEL_LOSSES.keys())}")
        loss_class = PIXEL_LOSSES[loss_class.lower()]
    elif loss_class is None:
        loss_class = nn.MSELoss  # default
        print("No loss_class provided, using default MSELoss.")

    params = loss_params.copy() if loss_params is not None else None

    # if the loss is masked, the reduction is handled in MaskedLoss
    if masked_loss and params is not None:
        # pop 'reduction' from loss_params and pass to MaskedLoss
        reduction = params.pop("reduction", "mean")
        criterion = MaskedLoss(loss_class(reduction="none", **params), reduction=reduction)
        print(f"Using masked loss: {loss_class.__name__} with params {params} and reduction {reduction}")
    elif masked_loss:
        criterion = MaskedLoss(loss_class(reduction="none"), reduction="mean")
        print(f"Using masked loss: {loss_class.__name__} with default params and reduction 'mean'")
    else:
        if params is not None:
            criterion = loss_class(**params)
            print(f"Using custom loss: {loss_class.__name__} with params {params}")
        else:
            criterion = loss_class()
            print(f"Using loss: {loss_class.__name__} with default params")

    return criterion
