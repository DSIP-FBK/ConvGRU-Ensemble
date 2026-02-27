<div align="center">

# ConvGRU-Ensemble

**Ensemble precipitation nowcasting using Convolutional GRU networks**

*Pretrained model for Italy:* ***IRENE*** — **I**talian **R**adar **E**nsemble **N**owcasting **E**xperiment

[![CI](https://github.com/DSIP-FBK/ConvGRU-Ensemble/actions/workflows/ci.yml/badge.svg)](https://github.com/DSIP-FBK/ConvGRU-Ensemble/actions)
[![License: BSD-2](https://img.shields.io/badge/license-BSD--2-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/it4lia/irene)

<br>

<table align="center" style="border: none; border-collapse: collapse;">
<tr style="border: none;">
<td colspan="2" align="center" style="border: none; padding: 10px;"><a href="https://it4lia-aifactory.eu"><img src="https://it4lia-aifactory.eu/wp-content/uploads/2025/05/logo-IT4LIA-AI-factory.svg" width="200" alt="IT4LIA AI-Factory"></a></td>
</tr>
<tr style="border: none;">
<td align="center" style="border: none; padding: 10px;"><a href="https://www.fbk.eu"><img src="https://webvalley.fbk.eu/static/img/logos/fbk-logo-blue.png" width="120" alt="Fondazione Bruno Kessler"></a></td>
<td align="center" style="border: none; padding: 10px;"><a href="https://www.italiameteo.eu"><img src="https://it4lia-aifactory.eu/wp-content/uploads/2025/08/logo-italiameteo.svg" width="180" alt="ItaliaMeteo"></a></td>
</tr>
</table>

<br>

The model encodes past radar frames into multi-scale hidden states and decodes them into an **ensemble of probabilistic forecasts** by running the decoder multiple times with different noise inputs, trained with **CRPS loss**.

</div>

---

## Quick Start

<details open>
<summary><b>Load from HuggingFace Hub</b></summary>

```python
from convgru_ensemble import RadarLightningModel

model = RadarLightningModel.from_pretrained("it4lia/irene")

import numpy as np
past = np.load("past_radar.npy")  # rain rate in mm/h, shape (T_past, H, W)
forecasts = model.predict(past, forecast_steps=12, ensemble_size=10)
# forecasts.shape = (10, 12, H, W) — 10 members, 12 future steps, mm/h
```

</details>

<details>
<summary><b>CLI Inference</b></summary>

```bash
convgru-ensemble predict \
    --input examples/sample_data.nc \
    --hub-repo it4lia/irene \
    --forecast-steps 12 \
    --ensemble-size 10 \
    --output predictions.nc
```

</details>

<details>
<summary><b>Serve via API</b></summary>

```bash
# With Docker
docker compose up

# Or directly
pip install convgru-ensemble[serve]
convgru-ensemble serve --hub-repo it4lia/irene --port 8000
```

**Submit a forecast request:**

```bash
# 4-hour forecast (4 steps × 1h) with 5 ensemble members
curl -X POST "http://localhost:8000/predict?forecast_steps=4&ensemble_size=5" \
    -F "file=@examples/sample_data.nc" \
    -o predictions.nc

# Use default settings (12 steps, 10 members)
curl -X POST http://localhost:8000/predict \
    -F "file=@examples/sample_data.nc" -o predictions.nc
```

**Read the predictions:**

```python
import xarray as xr

ds = xr.open_dataset("predictions.nc")
print(ds.precipitation_forecast.shape)
# (5, 4, 1400, 1200) — ensemble_member, forecast_step, y, x
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata and hyperparameters |
| `/predict` | POST | Upload NetCDF, get ensemble forecast as NetCDF |

**`/predict` query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `variable` | `RR` | Name of the rain rate variable in the NetCDF |
| `forecast_steps` | `12` | Number of future 5-min steps (1–48, i.e. max 4h) |
| `ensemble_size` | `10` | Number of ensemble members (1–10) |

The input NetCDF must contain a 3D variable `(T, H, W)` with rain rate in mm/h and at least 2 timesteps.

</details>

<details>
<summary><b>Fine-tune on your data</b></summary>

```bash
pip install convgru-ensemble
# See "Training" section below
```

</details>

## Setup

Requires Python >= 3.13. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync                    # core dependencies
uv sync --extra serve      # + FastAPI serving
```

## Data Preparation

The training pipeline expects a Zarr dataset with a rain rate variable `RR` indexed by `(time, x, y)`.

<details>
<summary><b>1. Filter valid datacubes</b></summary>

Scan the Zarr and find all space-time datacubes with fewer than `n_nan` NaN values:

```bash
cd importance_sampler
uv run python filter_nan.py path/to/dataset.zarr \
    --start_date 2021-01-01 --end_date 2025-12-11 \
    --Dt 24 --w 256 --h 256 \
    --step_T 3 --step_X 16 --step_Y 16 \
    --n_nan 10000 --n_workers 8
```

</details>

<details>
<summary><b>2. Importance sampling</b></summary>

Sample valid datacubes with higher probability for rainier events:

```bash
uv run python sample_valid_datacubes.py path/to/dataset.zarr valid_datacubes_*.csv \
    --q_min 1e-4 --m 0.1 --n_workers 8
```

A pre-sampled CSV is provided in [`importance_sampler/output/`](importance_sampler/output/).

</details>

## Training

Training is configured via [Fiddle](https://github.com/google/fiddle). Run with defaults:

```bash
uv run python -m convgru_ensemble.train
```

Override parameters from the command line:

```bash
uv run python -m convgru_ensemble.train \
    --config config:experiment \
    --config set:model.num_blocks=5 \
    --config set:model.forecast_steps=12 \
    --config set:model.loss_class=crps \
    --config set:model.ensemble_size=2 \
    --config set:datamodule.batch_size=16 \
    --config set:trainer.max_epochs=100
```

Monitor with TensorBoard: `uv run tensorboard --logdir logs/`

| Parameter | Description | Default |
|---|---|---|
| `model.num_blocks` | Encoder/decoder depth | `5` |
| `model.forecast_steps` | Future steps to predict | `12` |
| `model.ensemble_size` | Ensemble members during training | `2` |
| `model.loss_class` | Loss function (`mse`, `mae`, `crps`, `afcrps`) | `crps` |
| `model.masked_loss` | Mask NaN regions in loss | `True` |
| `datamodule.steps` | Total timesteps per sample (past + future) | `18` |
| `datamodule.batch_size` | Batch size | `16` |

## Architecture

```
Input (B, T_past, 1, H, W)
    |
    v
+--------------------------+
|        Encoder           |  ConvGRU + PixelUnshuffle (x num_blocks)
|  Spatial dims halve at   |  Channels: 1 -> 4 -> 16 -> 64 -> 256 -> 1024
|  each block              |
+----------+---------------+
           | hidden states
           v
+--------------------------+
|        Decoder           |  ConvGRU + PixelShuffle (x num_blocks)
|  Noise input (x M runs)  |  Each run produces one ensemble member
|  for ensemble generation |
+----------+---------------+
           |
           v
Output (B, T_future, M, H, W)
```

## Docker

```bash
docker build -t convgru-ensemble .

# Run with local checkpoint
docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints \
    -e MODEL_CHECKPOINT=/app/checkpoints/model.ckpt convgru-ensemble

# Run with HuggingFace Hub
docker run -p 8000:8000 -e HF_REPO_ID=it4lia/irene convgru-ensemble
```

## Project Structure

```
ConvGRU-Ensemble/
+-- convgru_ensemble/          # Python package
|   +-- model.py               # ConvGRU encoder-decoder architecture
|   +-- losses.py              # CRPS, afCRPS, masked loss wrappers
|   +-- lightning_model.py     # PyTorch Lightning training module
|   +-- datamodule.py          # Dataset and data loading
|   +-- train.py               # Training entry point (Fiddle config)
|   +-- utils.py               # Rain rate <-> reflectivity conversions
|   +-- hub.py                 # HuggingFace Hub upload/download
|   +-- cli.py                 # CLI for inference and serving
|   +-- serve.py               # FastAPI inference server
+-- examples/                  # Sample data for testing
+-- importance_sampler/        # Data preparation scripts
+-- notebooks/                 # Example notebooks
+-- scripts/                   # Utility scripts (e.g., upload to Hub)
+-- tests/                     # Test suite
+-- Dockerfile                 # Container for serving API
+-- MODEL_CARD.md              # HuggingFace model card template
```

## Acknowledgements

<div align="center">

Developed at **Fondazione Bruno Kessler (FBK)**, Trento, Italy, as part of the **Italian AI-Factory (IT4LIA)**, an EU-funded initiative supporting AI adoption across SMEs, academia, and public/private sectors. This work showcases capabilities in the **Earth (weather and climate) vertical domain**.

<br>

<table align="center" style="border: none; border-collapse: collapse;">
<tr style="border: none;">
<td colspan="2" align="center" style="border: none; padding: 8px;"><a href="https://it4lia-aifactory.eu"><img src="https://it4lia-aifactory.eu/wp-content/uploads/2025/05/logo-IT4LIA-AI-factory.svg" width="170" alt="IT4LIA AI-Factory"></a></td>
</tr>
<tr style="border: none;">
<td align="center" style="border: none; padding: 8px;"><a href="https://www.fbk.eu"><img src="https://webvalley.fbk.eu/static/img/logos/fbk-logo-blue.png" width="100" alt="Fondazione Bruno Kessler"></a></td>
<td align="center" style="border: none; padding: 8px;"><a href="https://www.italiameteo.eu"><img src="https://it4lia-aifactory.eu/wp-content/uploads/2025/08/logo-italiameteo.svg" width="150" alt="ItaliaMeteo"></a></td>
</tr>
</table>

</div>

## License

BSD 2-Clause — see [LICENSE](LICENSE).
