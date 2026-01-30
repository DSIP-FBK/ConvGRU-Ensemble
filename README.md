# ConvGRU-Ensemble-Nowcasting
Convolutional GRU nowcasting model with probabilistic ensemble training with CPRS loss

This repository contains the code used for preparing the data and training the ConvGRU model. An importance sampling method is used to select the precipitation events on which the model is trained. The model is trained using a CRPS loss to generate an ensemble of forecasts. 

The repository also contains a pre-trained ConvGRU model.

# Sructure of the repository

The repository is structured as follows:
- `data/`: Contains the script to download the italian radar dataset.
- `convgru-ens/`: Contains the ConvGRU model architecture, the loss functions, the datamodule and the code to launch the training.
- `convgru-ens/checkpoints/`: Contains the pre-trained ConvGRU model.
- `convgru-ens/importance_sampler/`: Contains the importance sampling code.



# Setup

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies. Since the project already includes `pyproject.toml` and `uv.lock`, you can set up the environment with a single command:

```bash
uv sync
```

This will create a virtual environment and install all dependencies. To run scripts within this environment, use:

```bash
uv run <command>
```

# How to download the data

To download the italian radar dataset, we can use the script download_italian_radar_zarr.py.

```bash
uv run python data/download_italian_radar_zarr.py
```

The dataset will be downloaded in the `data/italian-radar-dpc-sri.zarr` directory. The zarr dataset contains radar data from YYYY-MM-DD to YYYY-MM-DD with a 10 minute time step, and from 2021-01-01 to 2025-12-11, with  a 5 minute time step. The dataset is updated with new radar data every 5 minutes. The total size of the dataset is > 55 GB.

# How to prepare the data

To prepare the data for the training, we first need to filter the datacubes that contain more than N_nan NaN values and to sample the resulting valid datacubes using importance sampler script.

The file filter_nan.py is used to filter the datacubes that contain more than N_nan NaN values. 
The arguments are:
- zarr_path: path to the zarr dataset
- start_date: start date
- end_date: end date
- Dt: time depth of the datacube
- w: x width of the datacube
- h: y height of the datacube
- step_T: time step of the moving window
- step_X: x step of the moving window
- step_Y: y step of the moving window
- n_workers: number of parallel workers
- n_nan: maximum number of NaNs per datacube

The file sample_valid_datacubes.py is used to sample the valid datacubes.
The arguments are:
- zarr_path: path to the zarr dataset
- csv_path: path to the csv with the valid datacube coordinates (created by filter_nan.py)
- q_min: minimum selection probability (default 1e-4)
- s: denominator in the exponential (default 1)
- m: factor weighting the mean rescaled rain rate (default 0.1)
- n_workers: number of parallel workers (default 8)
- n_rand: number of random sampling of each datacube (default 1)

The output of the script is a csv file with the coordinates of the valid datacubes and a metadata json file. The csv file is used as input for the training script. The metadata json file is used to save the hyperparameters of the importance sampler. A csv with pre-sampled datacubes is in `importance_sampler/output`

# How to train the model

After setting the hyperparameters in the train.py script, we can train the model

```bash
uv run python train.py
```

During the training, the metrics and some images of the predictions are saved in the logs directory. tensorboard can be used to visualize the metrics. After each epoch, the best model is saved in the checkpoints directory.

# How to use the model