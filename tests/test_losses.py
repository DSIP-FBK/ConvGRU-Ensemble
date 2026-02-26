import torch

from convgru_ensemble.losses import CRPS, MaskedLoss, build_loss


def test_crps_reduces_to_scalar():
    loss_fn = CRPS(reduction="mean")
    preds = torch.randn(2, 4, 5, 8, 8)  # (B, T, M, H, W)
    target = torch.randn(2, 4, 1, 8, 8)  # (B, T, 1, H, W)
    loss = loss_fn(preds, target)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0 or loss.item() == 0


def test_masked_loss_ignores_masked_pixels():
    base_loss = torch.nn.MSELoss(reduction="none")
    loss_fn = MaskedLoss(base_loss, reduction="mean")
    preds = torch.ones(1, 2, 1, 4, 4)
    target = torch.zeros(1, 2, 1, 4, 4)
    # Mask out everything — loss should be 0
    mask = torch.zeros(1, 1, 1, 4, 4)
    loss = loss_fn(preds, target, mask)
    assert loss.item() == 0.0


def test_build_loss_by_name():
    criterion = build_loss("crps", loss_params=None, masked_loss=False)
    assert isinstance(criterion, CRPS)


def test_build_loss_masked():
    criterion = build_loss("mse", loss_params=None, masked_loss=True)
    assert isinstance(criterion, MaskedLoss)
