"""Command-line interface for ConvGRU-Ensemble inference and serving."""

import time

import fire
import numpy as np
import xarray as xr


def _load_model(checkpoint: str | None = None, hub_repo: str | None = None, device: str = "cpu"):
    """Load model from local checkpoint or HuggingFace Hub."""
    from .lightning_model import RadarLightningModel

    if hub_repo is not None:
        print(f"Loading model from HuggingFace Hub: {hub_repo}")
        return RadarLightningModel.from_pretrained(hub_repo, device=device)
    elif checkpoint is not None:
        print(f"Loading model from checkpoint: {checkpoint}")
        return RadarLightningModel.from_checkpoint(checkpoint, device=device)
    else:
        raise ValueError("Either --checkpoint or --hub-repo must be provided.")


def predict(
    input: str,
    checkpoint: str | None = None,
    hub_repo: str | None = None,
    variable: str = "RR",
    forecast_steps: int = 12,
    ensemble_size: int = 10,
    device: str = "cpu",
    output: str = "predictions.nc",
):
    """
    Run inference on a NetCDF input file and save predictions as NetCDF.

    Args:
        input: Path to input NetCDF file with rain rate data (T, H, W) or (T, Y, X).
        checkpoint: Path to local .ckpt checkpoint file.
        hub_repo: HuggingFace Hub repo ID (e.g., 'it4lia/irene'). Alternative to --checkpoint.
        variable: Name of the rain rate variable in the NetCDF file.
        forecast_steps: Number of future timesteps to forecast.
        ensemble_size: Number of ensemble members to generate.
        device: Device for inference ('cpu' or 'cuda').
        output: Path for the output NetCDF file.
    """
    model = _load_model(checkpoint, hub_repo, device)

    # Load input data
    print(f"Loading input: {input}")
    ds = xr.open_dataset(input)
    if variable not in ds:
        available = list(ds.data_vars)
        raise ValueError(f"Variable '{variable}' not found. Available: {available}")

    data = ds[variable].values  # (T, H, W) or similar
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data (T, H, W), got shape {data.shape}")

    print(f"Input shape: {data.shape}")
    past = data.astype(np.float32)

    # Run inference
    t0 = time.perf_counter()
    preds = model.predict(past, forecast_steps=forecast_steps, ensemble_size=ensemble_size)
    elapsed = time.perf_counter() - t0
    print(f"Output shape: {preds.shape} (ensemble, time, H, W)")
    print(f"Elapsed: {elapsed:.2f}s")

    # Build output dataset
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
            "source_file": str(input),
        },
    )

    ds_out.to_netcdf(output)
    print(f"Predictions saved to: {output}")


def serve(
    checkpoint: str | None = None,
    hub_repo: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cpu",
):
    """
    Start the FastAPI inference server.

    Args:
        checkpoint: Path to local .ckpt checkpoint file.
        hub_repo: HuggingFace Hub repo ID (e.g., 'it4lia/irene'). Alternative to --checkpoint.
        host: Host to bind to.
        port: Port to listen on.
        device: Device for inference ('cpu' or 'cuda').
    """
    import os

    if checkpoint is not None:
        os.environ["MODEL_CHECKPOINT"] = checkpoint
    if hub_repo is not None:
        os.environ["HF_REPO_ID"] = hub_repo
    os.environ.setdefault("DEVICE", device)

    import uvicorn

    uvicorn.run("convgru_ensemble.serve:app", host=host, port=port)


def main():
    fire.Fire({"predict": predict, "serve": serve})


if __name__ == "__main__":
    main()
