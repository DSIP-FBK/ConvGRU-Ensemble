# ConvGRU-Ensemble

Ensemble precipitation nowcasting using Convolutional GRU networks trained with CRPS loss.

The model encodes past radar frames into multi-scale hidden states and decodes them into an ensemble of probabilistic forecasts by running the decoder multiple times with different noise inputs.

## Setup

Requires Python ≥ 3.13. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

## Quick start

### Inference with a pre-trained model

```python
import numpy as np
import xarray as xr
from lightning_model import RadarLightningModel

# Load model from checkpoint
model = RadarLightningModel.from_checkpoint("checkpoints/ConvGRU-CRPS_6past_12fut.ckpt")

# Load radar data (rain rate in mm/h, shape: (T, H, W))
radar = xr.open_dataarray("data/test_radar_sample_54.nc")
past = radar[:6].values  # 6 past frames (~30 min at 5-min resolution)

# Generate ensemble forecast
preds = model.predict(past, forecast_steps=12, ensemble_size=10)
# preds shape: (ensemble_size, forecast_steps, H, W) — rain rate in mm/h

ensemble_mean = np.nanmean(preds, axis=0)
```

See [`convgru-ens/notebooks/test_pretrained_model.ipynb`](convgru-ens/notebooks/test_pretrained_model.ipynb) for a full example with visualizations.

## Data preparation

The training pipeline expects a Zarr dataset with a rain rate variable `RR` indexed by `(time, x, y)`. The data preparation has two steps:

### 1. Filter valid datacubes

Scan the Zarr and find all space-time datacubes with fewer than `n_nan` NaN values:

```bash
cd convgru-ens/importance_sampler

uv run python filter_nan.py path/to/dataset.zarr \
    --start_date 2021-01-01 \
    --end_date 2025-12-11 \
    --Dt 24 --w 256 --h 256 \
    --step_T 3 --step_X 16 --step_Y 16 \
    --n_nan 10000 \
    --n_workers 8
```

This outputs a CSV of valid `(t, x, y)` coordinates.

### 2. Importance sampling

Sample the valid datacubes with higher probability for rainier events:

```bash
uv run python sample_valid_datacubes.py path/to/dataset.zarr valid_datacubes_*.csv \
    --q_min 1e-4 \
    --m 0.1 \
    --n_workers 8
```

This outputs a sampled CSV (used for training) and a metadata JSON. A pre-sampled CSV is provided in [`convgru-ens/importance_sampler/output/`](convgru-ens/importance_sampler/output/).

## Training

Training is configured via [Fiddle](https://github.com/google/fiddle). The default configuration is defined in `train.py:experiment()`. Run with defaults:

```bash
cd convgru-ens
uv run python train.py
```

Override any parameter from the command line:

```bash
uv run python train.py \
    --config config:experiment \
    --config set:model.num_blocks=5 \
    --config set:model.forecast_steps=12 \
    --config set:model.loss_class=crps \
    --config set:model.ensemble_size=2 \
    --config set:model.masked_loss=True \
    --config set:datamodule.batch_size=16 \
    --config set:datamodule.steps=18 \
    --config set:trainer.max_epochs=100
```

Export the config to YAML for inspection:

```bash
uv run python train.py --export_yaml config.yaml
```

Logs and checkpoints are saved under `logs/`. Monitor training with TensorBoard:

```bash
uv run tensorboard --logdir logs/
```

### Key training parameters

| Parameter | Description | Default |
|---|---|---|
| `model.input_channels` | Channels per grid point | `1` |
| `model.num_blocks` | Encoder/decoder depth | `5` |
| `model.forecast_steps` | Future steps to predict | `12` |
| `model.ensemble_size` | Ensemble members | `2` |
| `model.loss_class` | Loss function (`mse`, `mae`, `crps`, `afcrps`) | `crps` |
| `model.masked_loss` | Mask out NaN regions | `True` |
| `datamodule.steps` | Total timesteps per sample (past + future) | `18` |
| `datamodule.batch_size` | Batch size | `16` |

## Architecture

```
Input (B, T_past, 1, H, W)
    │
    ▼
┌─────────────────────────┐
│        Encoder           │  ConvGRU + PixelUnshuffle (×num_blocks)
│  Spatial dims halve at   │  Channels: 1 → 4 → 16 → 64 → 256 → 1024
│  each block              │
└─────────┬───────────────┘
          │ hidden states
          ▼
┌─────────────────────────┐
│        Decoder           │  ConvGRU + PixelShuffle (×num_blocks)
│  Noise input (×M runs)   │  Each run produces one ensemble member
│  for ensemble generation │
└─────────┬───────────────┘
          │
          ▼
Output (B, T_future, M, H, W)
```

## Project structure

```
convgru-ens/
├── model.py              # ConvGRU encoder-decoder architecture
├── losses.py             # CRPS, afCRPS, masked loss wrappers
├── lightning_model.py    # PyTorch Lightning training module
├── datamodule.py         # Dataset and data loading
├── train.py              # Training entry point (Fiddle config)
├── utils.py              # Rain rate ↔ reflectivity conversions
├── importance_sampler/   # Data preparation scripts
│   ├── filter_nan.py
│   ├── sample_valid_datacubes.py
│   └── output/           # Pre-sampled datacube coordinates
└── notebooks/
    └── test_pretrained_model.ipynb
```

## License

BSD 2-Clause — see [LICENSE](LICENSE).
