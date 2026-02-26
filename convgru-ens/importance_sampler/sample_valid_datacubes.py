import argparse
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

START = time.time()
SEED = None  # for reproducibility

# === Parse Arguments ===
parser = argparse.ArgumentParser(description="Importance sampler of the valid datacubes (after the nan filtering)")
parser.add_argument("zarr_path", help="Path to the Zarr dataset")
parser.add_argument(
    "csv_path", help="Path to the CSV with the valid datacube coordinates (created by the nan filtering)"
)
parser.add_argument("--q_min", type=float, default=1e-4, help="Minimum selection probability (default 1e-4)")
parser.add_argument("--s", type=float, default=1, help="Denominator in the exponential")
parser.add_argument("--m", type=float, default=0.1, help="Factor weighting the mean rescaled rain rate (dafault 0.1)")
parser.add_argument("--n_workers", type=int, default=8, help="Number of parallel workers (default 8)")
parser.add_argument("--n_rand", type=int, default=1, help="Number of random sampling of each datacube (dafaut 1)")
args = parser.parse_args()

# === PARAMETERS ===
s = args.s
qmin = args.q_min
m = args.m

n_workers = args.n_workers  # number of parallel workers
N_rand = args.n_rand  # number of random numbers per region
chunksize = 16000  # = 500 CSV lines per workers

# Parameters from CSV filename
name_arr = args.csv_path.split("_")
dates = name_arr[2]
start_date = "-".join(dates.split("-")[0:3])
end_date = "-".join(dates.split("-")[3:])
Dt, w, h = name_arr[3].split("x")
step_T, step_X, step_Y = name_arr[4].split("x")
N_nan = name_arr[5][:-4]

# Casting
Dt, w, h = int(Dt), int(w), int(h)
step_T, step_X, step_Y = int(step_T), int(step_X), int(step_Y)
N_nan = int(N_nan)


# === FUNCTIONS ===
def acceptance_probability(data):
    """
    Calculate the acceptance probability for importance sampling.

    The probability is ``min(1, q_min + m * mean(data))``, where ``q_min``
    and ``m`` are module-level parameters.

    Parameters
    ----------
    data : np.ndarray
        Rescaled rain rate data for a single datacube.

    Returns
    -------
    q : float
        Acceptance probability in ``[q_min, 1]``.
    """
    return min(1.0, qmin + m * np.nanmean(data))


def process_datacube(coord, RR, N_rand, seed, acceptance_probability):
    """
    Process a single space-time region for importance sampling.

    Loads the datacube, rescales rain rate, computes an acceptance
    probability, and performs ``N_rand`` random acceptance trials.

    Parameters
    ----------
    coord : array-like of int
        Three-element sequence ``(it, ix, iy)`` specifying the datacube
        origin.
    RR : xr.DataArray
        Rain rate data array from the Zarr dataset.
    N_rand : int
        Number of random acceptance trials per datacube.
    seed : int or None
        Random seed for reproducibility. If ``None``, non-deterministic.
    acceptance_probability : callable
        Function that takes a data array and returns a probability in
        ``[0, 1]``.

    Returns
    -------
    hits : list of tuple of int
        List of accepted ``(it, ix, iy)`` tuples (may contain duplicates
        if accepted multiple times).
    """

    try:
        it, ix, iy = coord
        time_slice = slice(it, it + Dt)
        x_slice = slice(ix, ix + w)
        y_slice = slice(iy, iy + h)

        # Load data from Zarr
        data = RR[time_slice, x_slice, y_slice]
        data = 1 - np.exp(-data / s)

        # Calculate acceptance probability
        q = acceptance_probability(data)

        # Generate random numbers with seed for reproducibility
        rng = np.random.default_rng(seed)
        random_numbers = rng.random(N_rand)
        accepted_count = np.sum(random_numbers <= q)

        # Return accepted hits
        hits = [(it, ix, iy)] * accepted_count
        return hits
    except Exception as e:
        print(f"Error processing region ({it}, {ix}, {iy}): {e}", file=sys.stderr)
        return []


def file_writer(output_queue, filename, batch_size=1000):
    """
    Dedicated writer thread that flushes results to a CSV file in batches.

    Reads lists of ``(t, x, y)`` tuples from the queue and writes them as
    CSV rows. Stops when a ``None`` sentinel is received.

    Parameters
    ----------
    output_queue : queue.Queue
        Thread-safe queue providing lists of ``(t, x, y)`` tuples.
    filename : str
        Path to the output CSV file.
    batch_size : int, optional
        Number of rows to buffer before flushing to disk. Default is
        ``1000``.
    """
    with open(filename, "w") as f:
        f.write("t,x,y\n")
        batch = []

        while True:
            item = output_queue.get()

            if item is None:  # Sentinel value to stop
                # Write remaining batch
                for t, x, y in batch:
                    f.write(f"{t},{x},{y}\n")
                break

            batch.extend(item)

            if len(batch) >= batch_size:
                for t, x, y in batch:
                    f.write(f"{t},{x},{y}\n")
                f.flush()
                batch = []

        print(f"Results saved to {filename}")


# === Dataset Loading ===
print(f"Opening Zarr dataset: {args.zarr_path}")
try:
    zg = xr.open_zarr(args.zarr_path, mode="r")
    RR = zg["RR"]
except Exception as e:
    print(f"Error loading Zarr dataset: {e}")
    sys.exit(1)

# Chek if file exists
output_file = f"sampled_datacubes_{start_date}-{end_date}_{Dt}x{w}x{h}_{step_T}x{step_X}x{step_Y}_{N_nan}.csv"
if os.path.exists(output_file):
    response = input(f"File {output_file} already exists. Overwrite? (y/n): ")
    if response.lower() != "y":
        print("Exiting without overwriting.")
        sys.exit(0)
    else:
        print(f"Overwriting {output_file}...")

# Start writer thread
output_queue = Queue(maxsize=100)
writer_thread = Thread(target=file_writer, args=(output_queue, output_file, 1000))
writer_thread.daemon = False
writer_thread.start()

# save metadata
metadata = {
    "csv": args.csv_path,
    "zarr": args.zarr_path,
    "file": output_file,
    "start_date": start_date,
    "end_date": end_date,
    "Dt": Dt,
    "w": w,
    "h": h,
    "step_T": step_T,
    "step_X": step_X,
    "step_Y": step_Y,
    "N_nan": N_nan,
    "N_rand": N_rand,
    "n_workers": n_workers,
    "qmin": qmin,
    "m": m,
    "s": s,
    "seed": SEED,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
metadata_filename = output_file.replace(".csv", "_metadata.json")
with open(metadata_filename, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved run metadata to {metadata_filename}")


# === IMPORTANCE SAMPLING ===
# Create partial function with fixed parameters
process_datacube_partial = partial(
    process_datacube, RR=RR, N_rand=N_rand, seed=SEED, acceptance_probability=acceptance_probability
)

pool_chunksize = max(1, chunksize // n_workers)

with Pool(n_workers) as pool:
    pbar = tqdm(desc="Processing CSV chunks")

    # Loading the CSV by chunks
    for chunk in pd.read_csv(
        args.csv_path,
        usecols=["t", "x", "y"],
        dtype={"t": "int32", "x": "int32", "y": "int32"},
        engine="c",
        chunksize=chunksize,
    ):
        for hits in pool.imap(process_datacube_partial, chunk.values, chunksize=pool_chunksize):
            if hits:
                output_queue.put(hits)
            pbar.update(1)

    pbar.close()

# Signal writer thread to stop
output_queue.put(None)
writer_thread.join()

print(f"Done in {time.time() - START}s.")
sys.exit(0)
