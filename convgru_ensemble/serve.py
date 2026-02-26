"""FastAPI inference server for ConvGRU-Ensemble nowcasting model."""

import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import xarray as xr
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import Response

_model = None


def _load_model():
    from .lightning_model import RadarLightningModel

    device = os.environ.get("DEVICE", "cpu")
    checkpoint = os.environ.get("MODEL_CHECKPOINT")
    hub_repo = os.environ.get("HF_REPO_ID")

    if hub_repo:
        return RadarLightningModel.from_pretrained(hub_repo, device=device)
    elif checkpoint:
        return RadarLightningModel.from_checkpoint(checkpoint, device=device)
    else:
        raise RuntimeError("Set MODEL_CHECKPOINT or HF_REPO_ID environment variable.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = _load_model()
    yield
    _model = None


app = FastAPI(
    title="ConvGRU-Ensemble Nowcasting API",
    version="0.1.0",
    description="Ensemble precipitation nowcasting from radar data",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info")
async def model_info():
    """Return model metadata."""
    if _model is None:
        return {"error": "Model not loaded"}
    hp = _model.hparams
    return {
        "architecture": "ConvGRU-Ensemble EncoderDecoder",
        "input_channels": hp.input_channels,
        "num_blocks": hp.num_blocks,
        "forecast_steps": hp.forecast_steps,
        "ensemble_size": hp.ensemble_size,
        "noisy_decoder": hp.noisy_decoder,
        "loss_class": str(hp.loss_class),
        "device": str(_model.device),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="NetCDF file with rain rate data (T, H, W)"),  # noqa: B008
    variable: str = Query("RR", description="Name of the rain rate variable"),  # noqa: B008
    forecast_steps: int = Query(12, ge=1, le=48, description="Number of future timesteps"),  # noqa: B008
    ensemble_size: int = Query(10, ge=1, le=50, description="Number of ensemble members"),  # noqa: B008
):
    """
    Run ensemble nowcasting inference on uploaded NetCDF data.

    Accepts a NetCDF file containing past radar rain rate observations and
    returns NetCDF predictions with ensemble forecasts.
    """
    t0 = time.perf_counter()

    # Read uploaded NetCDF
    content = await file.read()
    ds = xr.open_dataset(io.BytesIO(content))

    if variable not in ds:
        available = list(ds.data_vars)
        return Response(
            content=f"Variable '{variable}' not found. Available: {available}",
            status_code=422,
        )

    data = ds[variable].values
    if data.ndim != 3:
        return Response(
            content=f"Expected 3D data (T, H, W), got shape {data.shape}",
            status_code=422,
        )

    past = data.astype(np.float32)

    # Run inference
    preds = _model.predict(past, forecast_steps=forecast_steps, ensemble_size=ensemble_size)

    elapsed = time.perf_counter() - t0

    # Build output NetCDF
    ds_out = xr.Dataset(
        {
            "precipitation_forecast": xr.DataArray(
                data=preds,
                dims=["ensemble_member", "forecast_step", "y", "x"],
                attrs={"units": "mm/h", "long_name": "Ensemble precipitation forecast"},
            ),
        },
        attrs={
            "model": "ConvGRU-Ensemble",
            "forecast_steps": forecast_steps,
            "ensemble_size": ensemble_size,
            "elapsed_seconds": f"{elapsed:.3f}",
        },
    )

    buf = io.BytesIO()
    ds_out.to_netcdf(buf)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/x-netcdf",
        headers={
            "Content-Disposition": "attachment; filename=predictions.nc",
            "X-Elapsed-Seconds": f"{elapsed:.3f}",
        },
    )
