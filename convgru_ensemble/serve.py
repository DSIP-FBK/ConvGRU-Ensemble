"""FastAPI inference server for ConvGRU-Ensemble nowcasting model."""

import io
import os
import tempfile
import time
from contextlib import asynccontextmanager

import magic
import numpy as np
import xarray as xr
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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
    forecast_steps: int = Query(12, ge=1, le=48, description="Number of future 5-min steps (max 48 = 4h)"),  # noqa: B008
    ensemble_size: int = Query(10, ge=1, le=10, description="Number of ensemble members (max 10)"),  # noqa: B008
):
    """
    Run ensemble nowcasting inference on uploaded NetCDF data.

    Accepts a NetCDF file containing past radar rain rate observations and
    returns NetCDF predictions with ensemble forecasts.
    """
    t0 = time.perf_counter()

    # Read file and check size (max 100 MB)
    max_size = 100 * 1024 * 1024
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1024 / 1024:.0f} MB). Maximum is 100 MB.",
        )

    mime = magic.from_buffer(content, mime=True)
    if mime == "application/x-hdf5":
        engine = "h5netcdf"
    elif mime in ("application/x-netcdf", "application/octet-stream") and content[:3] == b"CDF":
        engine = "scipy"
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Expected a NetCDF/HDF5 file, got '{mime}'.",
        )
    try:
        ds = xr.open_dataset(io.BytesIO(content), engine=engine)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to read NetCDF file: {exc}",
        ) from exc

    # Check variable exists
    if variable not in ds:
        available = list(ds.data_vars)
        raise HTTPException(
            status_code=422,
            detail=f"Variable '{variable}' not found. Available: {available}",
        )

    da = ds[variable]

    # Must be 3D
    if da.ndim != 3:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 3D variable (time, y, x), got {da.ndim}D with dims {da.dims}.",
        )

    # First dimension must be temporal
    time_names = {"time", "t", "step", "forecast_time", "lead_time"}
    first_dim = da.dims[0].lower()
    if first_dim not in time_names and da.shape[0] >= da.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"First dimension should be time, got dims {da.dims} with shape {da.shape}. "
                "Expected shape (T, H, W) where T < H and T < W."
            ),
        )

    if da.shape[0] < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 2 timesteps, got {da.shape[0]}.",
        )

    data = da.values
    if np.isinf(data).any():
        raise HTTPException(
            status_code=422,
            detail="Input data contains Inf values.",
        )

    # Replace NaN with 0 (no rain) — common for masked radar pixels
    past = np.nan_to_num(data, nan=0.0).astype(np.float32)

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

    encoding = {
        "precipitation_forecast": {"zlib": True, "complevel": 4},
    }
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    ds_out.to_netcdf(tmp_out_path, engine="netcdf4", encoding=encoding)
    with open(tmp_out_path, "rb") as fh:
        out_bytes = fh.read()
    os.unlink(tmp_out_path)

    return Response(
        content=out_bytes,
        media_type="application/x-netcdf",
        headers={
            "Content-Disposition": "attachment; filename=predictions.nc",
            "X-Elapsed-Seconds": f"{elapsed:.3f}",
        },
    )
