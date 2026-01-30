import sys, os, time, argparse
import numpy as np
import pandas as pd
import zarr
from multiprocessing import Pool
from functools import partial
from queue import Queue
from threading import Thread
from tqdm import tqdm


START = time.time()

# === Parse Arguments ===
parser = argparse.ArgumentParser(description='Process valid datacubes from Zarr dataset')
parser.add_argument('zarr_path', help='Path to the Zarr dataset')
parser.add_argument('--start_date', default=None, type=str, help='Start date (YYYY-MM-DD)')
parser.add_argument('--end_date', default=None, type=str, help='End date (YYYY-MM-DD)')
parser.add_argument('--Dt', type=int, default=24, help='Time depth')
parser.add_argument('--w', type=int, default=256, help='Spatial width')
parser.add_argument('--h', type=int, default=256, help='Spatial height')
parser.add_argument('--step_T', type=int, default=3, help='Time step')
parser.add_argument('--step_X', type=int, default=16, help='X step')
parser.add_argument('--step_Y', type=int, default=16, help='Y step')
parser.add_argument('--n_workers', type=int, default=8, help='Number of parallel workers')
parser.add_argument('--n_nan', type=int, default=10000, help='Maximum NaNs per datacube')
args = parser.parse_args()


# === PARAMETERS ===
Dt = args.Dt      # time depth
w = args.w        # x width
h = args.h        # y height
step_T = args.step_T
step_X = args.step_X
step_Y = args.step_Y
N_nan = args.n_nan # maximum number of nans in each datacube

n_workers = args.n_workers
time_chunk_size = 3 * Dt


# === Dataset Loading ===
print(f"Opening Zarr dataset: {args.zarr_path}")
try:
    zg = zarr.open(args.zarr_path, mode='r')
    RR_full = zg['RR']
    time_array_full = pd.to_datetime(zg['time'][:])
    
    print(f"Full dataset shape: T={RR_full.shape[0]}, X={RR_full.shape[1]}, Y={RR_full.shape[2]}")
    print(f"Full dataset time range: {time_array_full[0]} to {time_array_full[-1]}")
except Exception as e:
    print(f"Error loading Zarr dataset: {e}")
    sys.exit(1)

# Filter the dates
start_date = pd.to_datetime(args.start_date) if args.start_date else time_array_full[0]
end_date = pd.to_datetime(args.end_date) if args.end_date else time_array_full[-1]

# Find indices corresponding to date range
mask = (time_array_full >= start_date) & (time_array_full <= end_date)
valid_indices = np.where(mask)[0]

if len(valid_indices) == 0:
    print(f"No data found between {start_date} and {end_date}")
    sys.exit(1)

t_start_idx = valid_indices[0]
t_end_idx = valid_indices[-1] + 1

# Slice the data
size_T = t_end_idx - t_start_idx 
size_X = RR_full.shape[1]
size_Y = RR_full.shape[2]
time_array = time_array_full[t_start_idx:t_end_idx]

print(f"Filtered dataset shape: T={size_T}, X={size_X}, Y={size_Y}")
print(f"Filtered dataset time range: {time_array[0]} to {time_array[-1]}")

# Calculate maximum valid indices
max_x = size_X - w + 1
max_y = size_Y - h + 1
max_t = size_T - Dt + 1


# === Functions ===
def dim_nan_count(mask, dim, delta, dim_len):
    """
    Compute the number of NaN in each delta along dim

    Args:
        mask: array of NaN
        dim: (int) dimension along which to compute NaN
        delta: (int) number of pixels along dim
        dim_len: (int) lenght of the dimension

    Returns
        array of integers with the number of NaN
    """
    cumsum = np.cumsum(mask, axis=dim, dtype=np.int32)

    # Pad with zeros at the start along 'dim'
    pad_width = [(1, 0) if i == dim else (0, 0) for i in range(3)]
    padded_cumsum = np.pad(cumsum, pad_width=pad_width, mode='constant', constant_values=0)

    # Rolling window: padded[start+delta:start+delta+dim_len] - padded[start:start+dim_len-delta]    
    slices_start = [slice(dim_len - delta) if i == dim else slice(None) for i in range(3)]
    slices_end = [slice(delta, dim_len) if i == dim else slice(None) for i in range(3)]
    
    # number of nans in each delta
    return padded_cumsum[tuple(slices_end)] - padded_cumsum[tuple(slices_start)]


def dc_nan_count(chunk, deltas, dim_lenghts):
    """
    Compute the number of nan in each datacube
    
    Args:
        chunk: time chunk of the total zarr (full space)
    
    Returns: 
        array of integers with the number of NaN for each datacube:
        nans_cube_chunk(it,ix,iy) = number of NaN in (it:it+Dt, ix:ix+w, iy:iy+h)
    """
    # Compute NaN mask and cumsum along time axis
    nan_mask = np.isnan(chunk).astype(np.int16)
    
    # Number of NaN along time
    nans_t = dim_nan_count(nan_mask, dim=0, delta=deltas[0], dim_len=dim_lenghts[0])

    # Number of NaN along X x T
    nans_xt = dim_nan_count(nans_t, dim=1, delta=deltas[1], dim_len=dim_lenghts[1])

    # Number of NaN in the datacube (Y x X x T)
    nans_cube_chunk = dim_nan_count(nans_xt, dim=2, delta=deltas[2], dim_len=dim_lenghts[2])
    
    return nans_cube_chunk


def process_chunk(time_range, t_start_idx, data, N_nan, deltas, steps, valid_starts_gap, dc_nan_count):
    """
    Process a single time chunk and return valid indices.
    
    Args:
        time_range: Start and end indices of the chunk
        t_start_idx: Index corresponding to start_date
        data: Zarr array
        N_nan: maximum number of NaN in each datacube
        Dt: time step along time

        ... other parameters ...
        dc_nan_count: Function to count NaNs in datacubes
    
    Returns:
        tuple of (idx_x, idx_y, idx_t)
    """
    try:
        # start_t: start index of the chunk
        start_t, end_t = time_range

        # Chunk from Zarr (T, X, Y)
        chunk = data[start_t + t_start_idx : end_t + t_start_idx, :, :]
        dim_lenghts = chunk.shape  # shape: (T, X, Y)

        # Compute the number of NaNs in each datacube in chunk
        nans_cube_chunk = dc_nan_count(chunk, deltas, dim_lenghts)
        del chunk 

        # Apply the mask
        valid_mask = nans_cube_chunk <= N_nan
        del nans_cube_chunk

        # This indices are relative to the chunk
        idx_t_rel, idx_x, idx_y = np.where(valid_mask)
        del valid_mask

        # Cast to int32
        idx_t_rel = idx_t_rel.astype(np.int32)
        idx_x = idx_x.astype(np.int32)
        idx_y = idx_y.astype(np.int32)
        
        # Convert relative time indices
        idx_t = idx_t_rel + start_t
        
        # Keep only time indices in valid_starts_gap
        time_mask = np.isin(idx_t, valid_starts_gap)
        idx_t = idx_t[time_mask] + t_start_idx  # also convert to absolute index
        idx_x = idx_x[time_mask]
        idx_y = idx_y[time_mask]        

        # Filter datacube indices according to steps
        stride_mask = (idx_t % steps[0] == 0) & (idx_x % steps[1] == 0) & (idx_y % steps[2] == 0)
        idx_x = idx_x[stride_mask]
        idx_y = idx_y[stride_mask]
        idx_t = idx_t[stride_mask]

        return idx_t, idx_x, idx_y
    
    except Exception as e:
        print(f"Error processing chunk starting at t={start_t}: {e}", file=sys.stderr)
        sys.exit(1)


def file_writer(output_queue, filename, batch_size=1000):
    """
    Dedicated thread that writes results to file as they arrive from queue.
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
            
            batch.extend(zip(*item))
            
            if len(batch) >= batch_size:
                for t, x, y in batch:
                    f.write(f"{t},{x},{y}\n")
                f.flush()
                batch = []
        
        print(f"Results saved to {filename}")


# === Time Continuity ===
print("Checking time continuity...")
try:
    expected_step = pd.Timedelta('00:05:00')
    time_diffs = time_array[1:] - time_array[:-1]
    gaps = (time_diffs != expected_step).astype(int)
    
    # Check continuity for windows of size Dt
    window_sum = np.convolve(gaps, np.ones(Dt - 1, dtype=int), mode='valid')
    
    # Find valid starting times: continuous windows at T_step intervals
    valid_starts_gap = np.where((window_sum == 0))[0]
    print(f"Found {len(valid_starts_gap)} valid time starts without gaps")

    
except Exception as e:
    print(f"Error in time continuity check: {e}", file=sys.stderr)
    sys.exit(1)


# === Chunked NaN Processing ===
# Memory per chunk (x 4 because float32)
estimated_chunk_memory_gb = (time_chunk_size * size_X * size_Y * 4) / (1024**3)
print(f"Estimated memory per chunk: {estimated_chunk_memory_gb:.2f} GB")
print(f"Estimated total memory: {(estimated_chunk_memory_gb * n_workers):.2f} GB")

# Process time in chunks with overlap Dt
t_starts = np.arange(0, max_t, time_chunk_size)
t_ends = np.minimum(t_starts + time_chunk_size + Dt - 1, size_T)
t_pairs = np.stack((t_starts, t_ends), axis=1)

# Create partial function with fixed parameters
process_chunk_partial = partial(
    process_chunk,
    t_start_idx=t_start_idx,
    data=RR_full,
    N_nan=N_nan,
    deltas=(Dt, w, h),
    steps=(step_T, step_X, step_Y),
    valid_starts_gap=valid_starts_gap,
    dc_nan_count=dc_nan_count
)

# Chek if file exists
output_file = f'valid_datacubes_{args.start_date}-{args.end_date}_{Dt}x{w}x{h}_{step_T}x{step_X}x{step_Y}_{N_nan}.csv'
if os.path.exists(output_file):
    response = input(f"File {output_file} already exists. Overwrite? (y/n): ")
    if response.lower() != 'y':
        print("Exiting without overwriting.")
        sys.exit(0)
    else:
        print(f"Overwriting {output_file}...")

# Start writer thread
output_queue = Queue(maxsize=100)
writer_thread = Thread(target=file_writer, args=(output_queue, output_file, 1000))
writer_thread.daemon = False
writer_thread.start()

# Process chunks in parallel
with Pool(n_workers) as pool:
    for i, hits in enumerate(tqdm(
        pool.imap(process_chunk_partial, t_pairs, chunksize=1),
        total=len(t_starts),
        desc="Processing time chunks"
    )):
        output_queue.put(hits)

# Signal writer thread to stop
output_queue.put(None)
writer_thread.join()

print(f'Done in {time.time() - START}s.')
sys.exit(0)