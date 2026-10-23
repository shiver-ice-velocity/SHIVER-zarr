from datetime import date
import glob
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np
import xarray as xr
import zarr

# --- 1. CONFIGURATION ---
PROMICE_DIR = (
    "/mnt/parscratch/users/gg1bjd/Data/Velocity/Greenland/PROMICE_edition5"
)
OUTPUT_ZARR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Greenland/multisource_zarr/greenland_multisource_velocity_spatial.zarr"
CATALOG_FILE = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Greenland/multisource_zarr/master_epoch_catalog_spatial.pkl"


# --- PROMICE Processing ---
def preprocess_promice(file_path):
    """Opens, formats, and converts units for a single PROMICE NetCDF file."""
    ds = xr.open_dataset(file_path, decode_times=True)

    # 1. Keep relevant variables
    vars_to_keep = [
        "land_ice_surface_velocity_magnitude",
        "land_ice_surface_velocity_magnitude_std",
        "land_ice_surface_easting_velocity",
        "land_ice_surface_northing_velocity",
        "land_ice_surface_easting_velocity_std",
        "land_ice_surface_northing_velocity_std",
        "crs",
        "time_bnds",
    ]
    ds = ds[vars_to_keep]

    # 2. Rename to match schema
    ds = ds.rename(
        {
            "land_ice_surface_velocity_magnitude": "speed",
            "land_ice_surface_velocity_magnitude_std": "speed_error",
            "land_ice_surface_easting_velocity": "vx",
            "land_ice_surface_northing_velocity": "vy",
            "land_ice_surface_easting_velocity_std": "vx_error",
            "land_ice_surface_northing_velocity_std": "vy_error",
            "crs": "spatial_ref",
        }
    )

    # 3. Convert units (m/day -> m/year), round to 1 decimal, set attributes
    vel_vars = ["speed", "speed_error", "vx", "vy", "vx_error", "vy_error"]
    for var in vel_vars:
        ds[var] = np.round(ds[var] * 365.25, 1)
        ds[var].attrs["units"] = "m/year"
        ds[var].attrs["grid_mapping"] = "spatial_ref"

    # 4. Add data_source identifier
    ds["data_source"] = xr.DataArray(
        np.array(["PROMICE"], dtype="<U50"), dims=["time"]
    )

    # 5. Fix time bounds dimensions and clean encoding
    if "time_bnds" in ds and ds["time_bnds"].dims == ("bnds", "time"):
        ds["time_bnds"] = ds["time_bnds"].transpose("time", "bnds")

    for var in ["time", "time_bnds"]:
        if var in ds:
            ds[var].encoding.clear()

    return ds


# --- IDENTIFY NEW FILES ---
def get_new_promice_files_from_pyramid():
    """Identifies new NetCDF files by checking Group 0 of the OME-Zarr store directly."""
    if not os.path.exists(OUTPUT_ZARR):
        print(f"Error: OME-Zarr store not found at {OUTPUT_ZARR}")
        sys.exit(1)

    ds_zarr = xr.open_zarr(OUTPUT_ZARR, group="0")

    promice_mask = ds_zarr["data_source"].values == "PROMICE"
    zarr_promice_times = ds_zarr["time"].values[promice_mask]

    all_nc_files = sorted(glob.glob(os.path.join(PROMICE_DIR, "*.nc")))
    new_files = []
    tolerance = np.timedelta64(2, "D")

    for f in all_nc_files:
        with xr.open_dataset(f, decode_times=True) as temp_ds:
            nc_time = temp_ds["time"].values[0]

        time_diffs = np.abs(zarr_promice_times - nc_time)
        if not np.any(time_diffs < tolerance):
            new_files.append(f)

    return new_files


# --- METADATA & VERSION MANAGEMENT ---
def bump_minor_version(version_str: str) -> str:
    """Increments the minor version number (e.g., '1.6.0' -> '1.7.0')."""
    try:
        parts = [int(p) for p in version_str.split(".")]
        if len(parts) == 3:
            return f"{parts[0]}.{parts[1] + 1}.0"
    except Exception:
        pass
    return version_str


def sync_zarr3_inline_metadata(
    zarr_path: str, bump_version: bool = True
) -> None:
    """Synchronizes Zarr v3 inline metadata across OME pyramid levels, bumps minor version, and sets last_updated."""
    zarr_dir = Path(zarr_path)
    root_json_path = zarr_dir / "zarr.json"

    if not root_json_path.exists():
        print(f"Warning: Root metadata file not found at {root_json_path}")
        return

    # 1. Update root metadata attributes (version and date)
    with open(root_json_path, "r") as f:
        root_meta = json.load(f)

    if "attributes" in root_meta and isinstance(root_meta["attributes"], dict):
        attrs = root_meta["attributes"]
        if bump_version and "dataset_version" in attrs:
            old_ver = attrs["dataset_version"]
            new_ver = bump_minor_version(old_ver)
            attrs["dataset_version"] = new_ver
            print(f"Dataset version bumped: {old_ver} -> {new_ver}")
        attrs["last_updated"] = date.today().isoformat()

    with open(root_json_path, "w") as f:
        json.dump(root_meta, f, indent=2)

    # 2. Fast walk to collect updated shapes from array nodes (skipping binary chunk directories)
    array_shapes = {}
    array_chunks = {}

    for root, dirs, files in os.walk(zarr_dir):
        # Prune binary chunk folders ('c', 'c0', etc.) to prevent filesystem freezes
        dirs[:] = [
            d
            for d in dirs
            if d != "c" and not (d.startswith("c") and d[1:].isdigit())
        ]
        for file in files:
            if file == "zarr.json" or file.endswith(".json"):
                json_path = Path(root) / file
                if json_path == root_json_path:
                    continue
                try:
                    with open(json_path, "r") as f:
                        meta = json.load(f)
                    if meta.get("node_type") == "array" and "shape" in meta:
                        rel_key = json_path.parent.relative_to(
                            zarr_dir
                        ).as_posix()
                        array_shapes[rel_key] = meta["shape"]
                        if "chunk_grid" in meta:
                            array_chunks[rel_key] = meta["chunk_grid"]
                except Exception:
                    pass

    # 3. Synchronize inline consolidated_metadata in root and level zarr.json files
    for root, dirs, files in os.walk(zarr_dir):
        dirs[:] = [
            d
            for d in dirs
            if d != "c" and not (d.startswith("c") and d[1:].isdigit())
        ]
        for file in files:
            if file == "zarr.json" or file.endswith(".json"):
                json_path = Path(root) / file
                with open(json_path, "r") as f:
                    meta = json.load(f)

                if "consolidated_metadata" in meta and isinstance(
                    meta["consolidated_metadata"], dict
                ):
                    inline_meta = meta["consolidated_metadata"].get(
                        "metadata", {}
                    )
                    modified = False

                    for key, var_meta in inline_meta.items():
                        matched_shape = array_shapes.get(key)
                        matched_chunk = array_chunks.get(key)

                        if (
                            matched_shape is not None
                            and "shape" in var_meta
                            and var_meta["shape"] != matched_shape
                        ):
                            var_meta["shape"] = matched_shape
                            modified = True
                        if (
                            matched_chunk is not None
                            and "chunk_grid" in var_meta
                            and var_meta["chunk_grid"] != matched_chunk
                        ):
                            var_meta["chunk_grid"] = matched_chunk
                            modified = True

                    if modified:
                        with open(json_path, "w") as f:
                            json.dump(meta, f, indent=2)

    print("OME-Zarr v3 inline metadata synchronized successfully.")


# --- MULTI-LEVEL APPEND LOOP ---
def append_new_data_to_pyramid(new_files):
    """Processes, downsamples, and appends new epochs to all pyramid levels."""
    if not new_files:
        print(
            "No new PROMICE files detected for OME-Zarr. Gracefully exiting."
        )
        sys.exit(0)

    print(
        f"Detected {len(new_files)} new PROMICE files. Beginning OME-Zarr append..."
    )

    # 1. Inspect existing Zarr store to identify pyramid levels
    z_store = zarr.open(OUTPUT_ZARR, mode="r")
    existing_levels = sorted(
        [int(k) for k in z_store.group_keys() if k.isdigit()]
    )
    print(f"Targeting existing Zarr pyramid levels: {existing_levels}")

    existing_epochs = []
    if os.path.exists(CATALOG_FILE):
        with open(CATALOG_FILE, "rb") as f:
            existing_epochs = pickle.load(f)

    # --- STEP A: Pre-process new files into Level 0 batch ---
    batch_datasets = []

    for i, f in enumerate(new_files):
        print(
            f"  [{i+1}/{len(new_files)}] Pre-processing {os.path.basename(f)}"
        )
        try:
            ds_slice = preprocess_promice(f)

            t = ds_slice["time"].values[0]
            tb = ds_slice["time_bnds"].values
            if ds_slice["time_bnds"].dims == ("bnds", "time"):
                tb = ds_slice["time_bnds"].transpose("time", "bnds").values[0]
            else:
                tb = tb[0] if tb.ndim > 1 else tb

            ds_slice = ds_slice.assign_coords(time=[t])
            ds_slice["time_bnds"] = (
                ["time", "bnds"],
                [tb] if tb.ndim == 1 else tb,
            )
            ds_slice["data_source"] = (
                ["time"],
                np.array(["PROMICE"], dtype="<U50"),
            )

            batch_datasets.append(ds_slice)
            existing_epochs.append(
                {"time": t, "time_bnds": tb, "source": "PROMICE", "path": f}
            )

        except Exception as e:
            print(f"Error pre-processing {f}: {e}")
            print(
                "Halting batch processing to prevent further store corruption."
            )
            break

    if not batch_datasets:
        print("No valid datasets were prepared for appending. Exiting.")
        sys.exit(1)

    print(
        f"Concatenating {len(batch_datasets)} files into a single batch for appending...",
        flush=True,
    )
    ds_batch = xr.concat(batch_datasets, dim="time")

    # Clear dataset attributes from ds_batch to prevent NetCDF metadata from polluting store attributes
    ds_batch.attrs.clear()

    # --- STEP B: Iterate through levels, coarsen batch, and append ---
    for level in existing_levels:
        print(f"Appending batch to Group '{level}'...", flush=True)

        if level > 0:
            ds_batch = ds_batch.coarsen(x=2, y=2, boundary="pad").mean(
                keep_attrs=True
            )
            ds_batch.attrs.clear()

        # Drop fixed spatial variables to avoid schema conflicts
        ds_level_write = ds_batch.drop_vars(
            ["x", "y", "spatial_ref"], errors="ignore"
        )

        for var in ds_level_write.variables:
            ds_level_write[var].encoding.clear()

        # Append batch directly to target pyramid level
        ds_level_write.to_zarr(
            OUTPUT_ZARR, group=str(level), append_dim="time", mode="a"
        )

    # Save epoch catalog
    with open(CATALOG_FILE, "wb") as f:
        pickle.dump(existing_epochs, f)

    # --- STEP C: Synchronize Zarr v3 Inline Metadata & Bump Version ---
    print(
        "Synchronizing OME-Zarr v3 inline metadata and bumping version...",
        flush=True,
    )
    sync_zarr3_inline_metadata(OUTPUT_ZARR, bump_version=True)

    print("Pyramid append process completed successfully!")


if __name__ == "__main__":
    new_files = get_new_promice_files_from_pyramid()
    append_new_data_to_pyramid(new_files)