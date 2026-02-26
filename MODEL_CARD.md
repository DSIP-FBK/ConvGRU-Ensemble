---
license: bsd-2-clause
language: en
tags:
  - weather
  - nowcasting
  - radar
  - precipitation
  - ensemble-forecasting
  - convgru
  - earth-observation
library_name: pytorch
pipeline_tag: image-to-image
---

# IRENE — Italian Radar Ensemble Nowcasting Experiment

**IRENE** is a ConvGRU encoder-decoder model for short-term precipitation forecasting (nowcasting) from radar data. The model generates probabilistic ensemble forecasts, producing multiple plausible future scenarios from a single input sequence.

## Model Description

- **Architecture**: ConvGRU encoder-decoder with PixelShuffle/PixelUnshuffle for spatial scaling
- **Input**: Sequence of past radar rain rate fields (T, H, W) in mm/h
- **Output**: Ensemble of future rain rate forecasts (E, T, H, W) in mm/h
- **Temporal resolution**: 5 minutes per timestep
- **Training loss**: Continuous Ranked Probability Score (CRPS) with temporal consistency regularization

The model encodes past radar observations into multi-scale hidden states using stacked ConvGRU blocks with PixelUnshuffle downsampling. The decoder generates forecasts by unrolling with different random noise inputs, producing diverse ensemble members that capture forecast uncertainty.

## Intended Uses

- Short-term precipitation forecasting (0-60 min ahead) from radar reflectivity data
- Probabilistic nowcasting with uncertainty quantification via ensemble spread
- Research on deep learning for weather prediction
- Fine-tuning on regional radar datasets

## How to Use

```python
from convgru_ensemble import RadarLightningModel

# Load from HuggingFace Hub
model = RadarLightningModel.from_pretrained("it4lia/irene")

# Run inference on past radar data (rain rate in mm/h)
import numpy as np
past = np.random.rand(6, 256, 256).astype(np.float32)  # 6 past timesteps
forecasts = model.predict(past, forecast_steps=12, ensemble_size=10)
# forecasts.shape = (10, 12, 256, 256) — 10 ensemble members, 12 future steps
```

## Training Data

Trained on the Italian DPC (Dipartimento della Protezione Civile) radar mosaic surface rain intensity (SRI) dataset, covering the Italian territory at ~1 km resolution with 5-minute temporal resolution.

## Training Procedure

- **Optimizer**: Adam (lr=1e-4)
- **Loss**: CRPS with temporal consistency penalty (lambda=0.01)
- **Batch size**: 16
- **Ensemble size during training**: 2 members
- **Input window**: 6 past timesteps (30 min)
- **Forecast horizon**: 12 future timesteps (60 min)
- **Data augmentation**: Random rotations and flips
- **NaN handling**: Masked loss for missing radar data

## Limitations

- Trained on Italian radar data; performance may degrade on other domains without fine-tuning
- 5-minute temporal resolution only
- Best suited for convective and stratiform precipitation; extreme events may be underrepresented
- Ensemble spread is generated via noisy decoder inputs, not a full Bayesian approach

## Acknowledgements

This model was developed as part of the **Italian AI-Factory** (IT4LIA), an EU-funded initiative supporting the adoption of AI across SMEs, academia, and public/private sectors. The AI-Factory provides free HPC compute, consultancy, and AI-ready datasets. This work showcases capabilities in the **Earth (weather and climate) vertical domain**.

Developed at **Fondazione Bruno Kessler (FBK)**, Trento, Italy.

## License

BSD 2-Clause License
