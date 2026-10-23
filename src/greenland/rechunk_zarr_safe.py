import json
import os
import shutil
import argparse
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster

def safe_p2p_rechunk(mode):
    # --- 1. SLURM & CLUSTER SETUP ---
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
    slurm_mem_mb = int(os.environ.get('SLURM_MEM_PER_NODE', 128000))
    
    cluster = LocalCluster(
        n_workers=slurm_cpus, 
        threads_per_worker=1,
        memory_limit=f"{int(slurm_mem_mb / slurm_cpus)}MB"
    )
    client = Client(cluster)
    dask.config.set({"array.rechunk.method": "p2p"})
    
    print(f"Dask cluster started with {slurm_cpus} workers.")
    print(f"P2P Shuffling Enabled. Dashboard: {client.dashboard_link}")

    # --- 2. PATH DEFINITIONS ---
    base_dir = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Greenland/multisource_zarr"
    source_path = os.path.join(base_dir, "greenland_multisource_velocity_spatial_200.zarr")
    
    if mode == "cubed":
        final_path = os.path.join(base_dir, "greenland_multisource_velocity_cubed.zarr")
    elif mode == "timeseries":
        final_path = os.path.join(base_dir, "greenland_multisource_velocity_timeseries.zarr")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    temp_target_path = final_path.replace(".zarr", "_temp.zarr")
    backup_path = final_path.replace(".zarr", "_backup.zarr")

    if os.path.exists(temp_target_path):
        print(f"Cleaning up orphaned temp directory: {temp_target_path}")
        shutil.rmtree(temp_target_path)

    # --- 3. DYNAMIC CHUNK DEFINITIONS ---
    print("Opening source Zarr...")
    ds = xr.open_zarr(source_path)

    # Clear stale chunk definitions from encoding while preserving codecs/compressors
    for var in ds.variables:
        ds[var].encoding.pop('chunks', None)
        ds[var].encoding.pop('preferred_chunks', None)

    time_len = ds.sizes['time']
    
    if mode == "cubed":
        chunk_dict = {'time': 300, 'y': 256, 'x': 256}
    elif mode == "timeseries":
        chunk_dict = {'time': time_len, 'y': 64, 'x': 64}

    print(f"Target Chunking for N-D data variables -> {chunk_dict}")

    # --- 4. EXECUTE P2P RECHUNK & PRESERVE ATTRIBUTES ---
    print("Applying chunk definitions...")
    ds_rechunked = ds.chunk(chunk_dict)
    
    # Enforce single-chunk (-1) format for 1D/0D coordinates, bounds, and scalar metadata arrays
    print("Enforcing single-chunk format for 0D/1D coordinate & metadata variables...")
    for var in ds_rechunked.variables:
        if len(ds_rechunked[var].dims) <= 1 or var in ['spatial_ref', 'crs', 'time_bnds', 'data_source']:
            ds_rechunked[var] = ds_rechunked[var].chunk(-1)

    # Inherit dataset attributes directly from spatial_200 source store
    # Option: log the rechunking mode in dataset metadata
    ds_rechunked.attrs = dict(ds.attrs)
    ds_rechunked.attrs["rechunk_mode"] = mode

    print("Writing to temporary Zarr v3 store...")
    ds_rechunked.to_zarr(temp_target_path, zarr_format=3, mode="w")

    # --- 5. ATOMIC SWAP ---
    print("Rechunking complete! Performing safe swap...")
    if os.path.exists(final_path):
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path) 
        os.rename(final_path, backup_path)
        
    os.rename(temp_target_path, final_path)
    
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)

    print(f"Success! '{mode.capitalize()}' Zarr is fully updated and ready at: {final_path}")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safely rechunk the spatially optimized Zarr store using Dask P2P.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["cubed", "timeseries"], 
        required=True,
        help="Choose the chunking schema: 'cubed' or 'timeseries'"
    )
    
    args = parser.parse_args()
    safe_p2p_rechunk(args.mode)