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

    for var in ds.variables:
        if 'chunks' in ds[var].encoding:
            del ds[var].encoding['chunks']

    time_len = ds.sizes['time']
    
    if mode == "cubed":
        chunk_dict = {'time': 300, 'y': 256, 'x': 256}
    elif mode == "timeseries":
        chunk_dict = {'time': time_len, 'y': 64, 'x': 64}

    print(f"Target Chunking -> {chunk_dict}")

    # --- 4. EXECUTE P2P RECHUNK & PRESERVE ATTRIBUTES ---
    print("Applying chunk definitions...")
    ds_rechunked = ds.chunk(chunk_dict)
    
    print("Enforcing single-chunk format for metadata arrays...")
    for var in ['time', 'time_bnds', 'data_source']:
        if var in ds_rechunked:
            ds_rechunked[var] = ds_rechunked[var].chunk(-1)

    # Preserve metadata attributes from destination store if present
    existing_attrs = {}
    if os.path.exists(final_path):
        final_json_path = os.path.join(final_path, "zarr.json")
        if os.path.exists(final_json_path):
            try:
                with open(final_json_path, "r") as f:
                    final_meta = json.load(f)
                    existing_attrs = final_meta.get("attributes", {})
            except Exception as e:
                print(f"Warning: Could not read existing attributes from {final_path}: {e}")

    # Merge: start with source attrs, update with existing target attrs
    merged_attrs = dict(ds.attrs)
    merged_attrs.update(existing_attrs)
    ds_rechunked.attrs = merged_attrs

    print("Writing to temporary store...")
    ds_rechunked.to_zarr(temp_target_path, zarr_format=3)

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