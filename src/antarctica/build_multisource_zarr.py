import xarray as xr
import numpy as np
import dask.array as da
import glob
import os
import sys
import zarr
import rioxarray
import pickle
from datetime import datetime, timedelta, date
import re

# --- Paths ---
ENVEO_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/ENVEO_monthly"
ITSLIVE_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/ITS_LIVE_annual"
MEASURES_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/MEaSUREs/annual"
MEASURES_MULTI_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/MEaSUREs/multiyear"
MEASURES_ASE_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/MEaSUREs/ASE"
SID_ANNUAL_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/SID/annual"
ESA_CCI_ANNUAL_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/ESA_CCI/annual"
JOUGHIN_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/Joughin/pine_island/RWArchive/velocityTimeSeries"
LI_TOTTEN_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/Totten"
ENVEO_S1_PIG_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/ENVEO_selected/pine_island/ais_cci_iv_PIG_S1t065_20141010_20190821_v1_1/mag"
ENVEO_OTHERS_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/ENVEO_selected/others"
SHIFT_DIR = "/mnt/parscratch/users/gg1bjd/SCADI/output/Sentinel1/Antarctica/mosaic/subregions/peninsula/date_pairs"
HISTORICAL_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/Historical"

OUTPUT_DIR = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/multisource_zarr"
OUTPUT_ZARR = os.path.join(OUTPUT_DIR, "antarctica_multisource_velocity_spatial_200.zarr")
CATALOG_FILE = os.path.join(OUTPUT_DIR, "master_epoch_catalog.pkl")

TARGET_VERSION = "1.5.0"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. ENVEO Monthly Processing ---
def parse_enveo_monthly_time(file_path):
    with xr.open_dataset(file_path) as ds:
        start_str = ds.attrs.get('time_coverage_start')[:10]
        end_str = ds.attrs.get('time_coverage_end')[:10]
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_enveo_monthly(file_path, master_x=None, master_y=None):
    ds_raw = xr.open_dataset(file_path)
    speed_raw = ds_raw['land_ice_surface_velocity_magnitude'] * 365.25
    vx_raw = ds_raw['land_ice_surface_easting_velocity'] * 365.25
    vy_raw = ds_raw['land_ice_surface_northing_velocity'] * 365.25
    vx_error_raw = ds_raw['land_ice_surface_easting_stddev'] * 365.25
    vy_error_raw = ds_raw['land_ice_surface_northing_stddev'] * 365.25
    speed_error_raw = np.sqrt(vx_error_raw**2 + vy_error_raw**2)
    # Interpolate the data
    if master_x is not None and master_y is not None:
        speed_interp = speed_raw.interp(x=master_x, y=master_y, method="nearest")
        vx_interp = vx_raw.interp(x=master_x, y=master_y, method="nearest")
        vy_interp = vy_raw.interp(x=master_x, y=master_y, method="nearest")
        speed_error_interp = speed_error_raw.interp(x=master_x, y=master_y, method="nearest")
        vx_error_interp = vx_error_raw.interp(x=master_x, y=master_y, method="nearest")
        vy_error_interp = vy_error_raw.interp(x=master_x, y=master_y, method="nearest")
        out_x, out_y = master_x.values, master_y.values
    else:
        speed_interp = speed_raw
        vx_interp = vx_raw
        vy_interp = vy_raw
        speed_error_interp = speed_error_raw
        vx_error_interp = vx_error_raw
        vy_error_interp = vy_error_raw
        out_x, out_y = ds_raw['x'].values, ds_raw['y'].values
    # Extract values and round    
    speed_vals = np.round(speed_interp.squeeze().values, 1)
    vx_vals = np.round(vx_interp.squeeze().values, 1)
    vy_vals = np.round(vy_interp.squeeze().values, 1)
    speed_error_vals = np.round(speed_error_interp.squeeze().values, 1)
    vx_error_vals = np.round(vx_error_interp.squeeze().values, 1)
    vy_error_vals = np.round(vy_error_interp.squeeze().values, 1)
    
    crs_val = ds_raw['crs'].values if 'crs' in ds_raw else 0
    crs_attrs = ds_raw['crs'].attrs if 'crs' in ds_raw else {}
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'spatial_ref': ([], crs_val, crs_attrs),
        'data_source': (['time'], np.array(["ENVEO_monthly"], dtype="<U50"))
    }, coords={'y': (['y'], out_y), 'x': (['x'], out_x)})
    return ds

# --- 2. ITS_LIVE Annual Processing ---
def parse_itslive_annual_time(file_path):
    match = re.search(r'_(\d{4})_', os.path.basename(file_path))
    year = int(match.group(1))
    start_dt = datetime(year, 1, 1)
    end_dt = datetime(year, 12, 31)
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_itslive_annual(file_path, master_x, master_y):
    ds_raw = xr.open_dataset(file_path)
    
    # Load raw data for all variables
    speed_raw = ds_raw['v']
    vx_raw = ds_raw['vx']
    vy_raw = ds_raw['vy']
    speed_error_raw = ds_raw['v_error']
    vx_error_raw = ds_raw['vx_error']
    vy_error_raw = ds_raw['vy_error']
    
    # Create valid mask based on speed and error bounds (vx and vy can legitimately be negative)
    valid_mask = (speed_raw >= 0) & (speed_raw != 32767) & (speed_error_raw >= 0) & (speed_error_raw != 32767)
    
    # Mask invalid data
    speed_masked = speed_raw.where(valid_mask, np.nan)
    vx_masked = vx_raw.where(valid_mask, np.nan)
    vy_masked = vy_raw.where(valid_mask, np.nan)
    speed_error_masked = speed_error_raw.where(valid_mask, np.nan)
    vx_error_masked = vx_error_raw.where(valid_mask, np.nan)
    vy_error_masked = vy_error_raw.where(valid_mask, np.nan)
    
    # Interpolate the masked data to the master grid
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_interp.squeeze().values, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_interp.squeeze().values, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_interp.squeeze().values, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_interp.squeeze().values, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_interp.squeeze().values, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_interp.squeeze().values, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["ITS_LIVE_annual"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 3. MEaSUREs Annual Processing ---
def parse_measures_annual_time(file_path):
    match = re.search(r'_(\d{4})_(\d{4})_', os.path.basename(file_path))
    start_dt = datetime(int(match.group(1)), 7, 1)
    end_dt = datetime(int(match.group(2)), 6, 30)
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_measures_annual(file_path, master_x, master_y):
    ds_raw = xr.open_dataset(file_path)
    
    # Extract raw variables
    vx_raw = ds_raw['VX']
    vy_raw = ds_raw['VY']
    vx_error_raw = ds_raw['ERRX']
    vy_error_raw = ds_raw['ERRY']
    
    speed_raw = np.sqrt(vx_raw**2 + vy_raw**2)
    speed_error_raw = np.sqrt(vx_error_raw**2 + vy_error_raw**2)
    
    # Mask invalid data (where speed == 0)
    valid_mask = (speed_raw != 0)
    speed_masked = speed_raw.where(valid_mask, np.nan)
    vx_masked = vx_raw.where(valid_mask, np.nan)
    vy_masked = vy_raw.where(valid_mask, np.nan)
    speed_error_masked = speed_error_raw.where(valid_mask, np.nan)
    vx_error_masked = vx_error_raw.where(valid_mask, np.nan)
    vy_error_masked = vy_error_raw.where(valid_mask, np.nan)
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.squeeze().values, 1)
    vx_vals = np.round(vx_interp.squeeze().values, 1)
    vy_vals = np.round(vy_interp.squeeze().values, 1)
    speed_error_vals = np.round(speed_error_interp.squeeze().values, 1)
    vx_error_vals = np.round(vx_error_interp.squeeze().values, 1)
    vy_error_vals = np.round(vy_error_interp.squeeze().values, 1)
    
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["MEaSUREs_annual"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 4. MEaSUREs Multiyear Processing ---
def parse_measures_multiyear_time(file_path):
    match = re.search(r'_(\d{4})-(\d{4})_', os.path.basename(file_path))
    start_dt = datetime(int(match.group(1)), 7, 1)
    end_dt = datetime(int(match.group(2)), 6, 30)
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_measures_multiyear(file_path, master_x, master_y):
    ds_raw = xr.open_dataset(file_path)
    
    # Extract raw variables
    vx_raw = ds_raw['VX']
    vy_raw = ds_raw['VY']
    vx_error_raw = ds_raw['ERRX']
    vy_error_raw = ds_raw['ERRY']
    
    speed_raw = np.sqrt(vx_raw**2 + vy_raw**2)
    speed_error_raw = np.sqrt(vx_error_raw**2 + vy_error_raw**2)
    
    # Mask invalid data (where speed == 0)
    valid_mask = (speed_raw != 0)
    speed_masked = speed_raw.where(valid_mask, np.nan)
    vx_masked = vx_raw.where(valid_mask, np.nan)
    vy_masked = vy_raw.where(valid_mask, np.nan)
    speed_error_masked = speed_error_raw.where(valid_mask, np.nan)
    vx_error_masked = vx_error_raw.where(valid_mask, np.nan)
    vy_error_masked = vy_error_raw.where(valid_mask, np.nan)
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.squeeze().values, 1)
    vx_vals = np.round(vx_interp.squeeze().values, 1)
    vy_vals = np.round(vy_interp.squeeze().values, 1)
    speed_error_vals = np.round(speed_error_interp.squeeze().values, 1)
    vx_error_vals = np.round(vx_error_interp.squeeze().values, 1)
    vy_error_vals = np.round(vy_error_interp.squeeze().values, 1)
    
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["MEaSUREs_multiyear"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 5. MEaSUREs ASE Processing ---
def catalog_measures_ase(file_path):
    epochs = []
    with xr.open_dataset(file_path) as ds:
        years = [int(var[2:]) for var in ds.data_vars if re.match(r'^vx\d{4}$', var)]
    for year in sorted(years):
        start_dt = datetime(year, 1, 1)
        end_dt = datetime(year, 12, 31)
        mid_dt = start_dt + (end_dt - start_dt) / 2
        t = np.datetime64(mid_dt)
        tb = np.array([np.datetime64(start_dt), np.datetime64(end_dt)])
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'MEaSUREs_ASE', 'path': file_path, 'year': year})
    return epochs

def preprocess_measures_ase(file_path, year, master_x, master_y):
    ds_raw = xr.open_dataset(file_path)
    
    # Rename coords and swap dims
    ds_renamed = ds_raw.assign_coords(x=ds_raw['xaxis'], y=ds_raw['yaxis']).swap_dims({'nx': 'x', 'ny': 'y'})
    
    # Extract raw variables (err is the speed error here)
    vx_raw = ds_renamed[f'vx{year}']
    vy_raw = ds_renamed[f'vy{year}']
    speed_error_raw = ds_renamed[f'err{year}']
    
    # Calculate speed
    speed_raw = np.sqrt(vx_raw**2 + vy_raw**2)
    
    # Mask invalid data
    valid_mask = (speed_raw > 0) & (speed_error_raw > 0)
    speed_masked = speed_raw.where(valid_mask, np.nan)
    vx_masked = vx_raw.where(valid_mask, np.nan)
    vy_masked = vy_raw.where(valid_mask, np.nan)
    speed_error_masked = speed_error_raw.where(valid_mask, np.nan)
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values as numpy arrays
    speed_vals = speed_interp.squeeze().values
    vx_vals = vx_interp.squeeze().values
    vy_vals = vy_interp.squeeze().values
    speed_error_vals = speed_error_interp.squeeze().values
    
    # Calculate vx_error and vy_error based on proportional contribution
    with np.errstate(divide='ignore', invalid='ignore'):
        vx_prop = np.where(speed_vals > 0, np.abs(vx_vals) / speed_vals, 0)
        vy_prop = np.where(speed_vals > 0, np.abs(vy_vals) / speed_vals, 0)
        
    vx_error_vals = speed_error_vals * vx_prop
    vy_error_vals = speed_error_vals * vy_prop
    
    # Round all values to 1 decimal place
    speed_vals = np.round(speed_vals, 1)
    vx_vals = np.round(vx_vals, 1)
    vy_vals = np.round(vy_vals, 1)
    speed_error_vals = np.round(speed_error_vals, 1)
    vx_error_vals = np.round(vx_error_vals, 1)
    vy_error_vals = np.round(vy_error_vals, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["MEaSUREs_ASE"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 6. SID Annual Processing ---
def parse_sid_annual_time(file_path):
    match = re.search(r'-(\d{4})-fv', os.path.basename(file_path))
    year = int(match.group(1))
    start_dt = datetime(year, 1, 1)
    end_dt = datetime(year, 12, 31)
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_sid_annual(file_path, master_x, master_y):
    ds_raw = xr.open_dataset(file_path)
    
    # Extract raw variables
    speed_raw = ds_raw['ice_speed'].squeeze()
    speed_error_raw = ds_raw['ice_speed_uncertainty'].squeeze()
    
    # Mask invalid data
    valid_mask = (speed_raw > 0) & (speed_error_raw > 0)
    speed_masked = speed_raw.where(valid_mask, np.nan)
    speed_error_masked = speed_error_raw.where(valid_mask, np.nan)
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    
    # Generate NaN arrays for missing components
    vx_vals = np.full_like(speed_vals, np.nan)
    vy_vals = np.full_like(speed_vals, np.nan)
    vx_error_vals = np.full_like(speed_vals, np.nan)
    vy_error_vals = np.full_like(speed_vals, np.nan)
    
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["SID_annual"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 7. ESA CCI Annual Processing ---
def parse_esacci_annual_time(file_path):
    match = re.search(r'_(\d{8})_(\d{8})_', os.path.basename(file_path))
    start_dt = datetime.strptime(match.group(1), "%Y%m%d")
    end_dt = datetime.strptime(match.group(2), "%Y%m%d")
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_esacci_annual(file_path, master_x, master_y):
    ds_raw = xr.open_dataset(file_path)
    
    # Extract raw variables and convert to m/year
    speed_raw = ds_raw['land_ice_surface_velocity_magnitude'] * 365.25
    vx_raw = ds_raw['land_ice_surface_easting_velocity'] * 365.25
    vy_raw = ds_raw['land_ice_surface_northing_velocity'] * 365.25
    vx_error_raw = ds_raw['land_ice_surface_easting_stddev'] * 365.25
    vy_error_raw = ds_raw['land_ice_surface_northing_stddev'] * 365.25
    
    # Calculate speed error from the scaled components
    speed_error_raw = np.sqrt(vx_error_raw**2 + vy_error_raw**2)
    
    # Mask invalid data
    valid_mask = (speed_raw > 0) & (speed_error_raw > 0)
    speed_masked = speed_raw.where(valid_mask, np.nan)
    vx_masked = vx_raw.where(valid_mask, np.nan)
    vy_masked = vy_raw.where(valid_mask, np.nan)
    speed_error_masked = speed_error_raw.where(valid_mask, np.nan)
    vx_error_masked = vx_error_raw.where(valid_mask, np.nan)
    vy_error_masked = vy_error_raw.where(valid_mask, np.nan)
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.squeeze().values, 1)
    vx_vals = np.round(vx_interp.squeeze().values, 1)
    vy_vals = np.round(vy_interp.squeeze().values, 1)
    speed_error_vals = np.round(speed_error_interp.squeeze().values, 1)
    vx_error_vals = np.round(vx_error_interp.squeeze().values, 1)
    vy_error_vals = np.round(vy_error_interp.squeeze().values, 1)
    
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["C3S_annual"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 8. Joughin Sentinel-1 Processing ---
def parse_joughin_s1_time(file_path):
    match = re.search(r'S1Quarterly\.(\d{4}-\d{2}-\d{2})\.', os.path.basename(file_path))
    mid_dt = datetime.strptime(match.group(1), "%Y-%m-%d")
    start_dt = mid_dt - timedelta(days=45)
    end_dt = mid_dt + timedelta(days=45)
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_joughin_s1(vx_path, master_x, master_y):
    vy_path = vx_path.replace('.vx.tif', '.vy.tif')
    
    # Load raw data
    vx_da = rioxarray.open_rasterio(vx_path).squeeze(drop=True)
    vy_da = rioxarray.open_rasterio(vy_path).squeeze(drop=True)
    
    # Calculate speed
    speed_raw = np.sqrt(vx_da**2 + vy_da**2)
    
    # Mask invalid data
    valid_mask = (vx_da > -200000) & (vx_da < 200000) & (vy_da > -200000) & (vy_da < 200000)
    speed_masked = speed_raw.where(valid_mask, np.nan)
    vx_masked = vx_da.where(valid_mask, np.nan)
    vy_masked = vy_da.where(valid_mask, np.nan)
    
    # Estimate errors as 5% of the absolute values
    speed_error_masked = speed_masked * 0.05
    vx_error_masked = np.abs(vx_masked) * 0.05
    vy_error_masked = np.abs(vy_masked) * 0.05
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["Joughin_Sentinel-1"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 9. Joughin TSX Processing ---
#def parse_joughin_tsx_time(file_path):
#    match = re.search(r'TSX\.(\d{4}-\d{2}-\d{2})\.', os.path.basename(file_path))
#    mid_dt = datetime.strptime(match.group(1), "%Y-%m-%d")
#    start_dt = mid_dt - timedelta(days=75)
#    end_dt = mid_dt + timedelta(days=75)
#    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

# Dictionary containing filename dates ('YYYY-MM-DD') and corresponding TSX (start, end) timestamps
TSX_METADATA_TABLE = {
    "2009-03-31": ("18/02/2009 09:43", "14/04/2009 09:44"),
    "2009-09-15": ("16/11/2009 09:18", "01/01/2010 09:44"),
    "2010-11-30": ("27/10/2010 09:44", "10/12/2010 09:44"),
    "2011-11-30": ("21/10/2011 09:18", "08/12/2011 09:44"),
    "2012-03-31": ("04/03/2012 03:47", "28/04/2012 03:47"),
    "2012-07-01": ("10/06/2012 04:04", "24/08/2012 09:18"),
    "2013-06-01": ("28/05/2013 04:04", "12/07/2013 03:47"),
    "2014-01-30": ("20/11/2013 04:04", "12/03/2014 09:44"),
    "2014-08-01": ("07/06/2014 03:47", "03/09/2014 03:47"),
    "2015-01-01": ("10/12/2014 04:04", "26/02/2015 03:47"),
    "2015-06-01": ("19/05/2015 03:56", "04/08/2015 03:56"),
    "2015-09-15": ("01/09/2015 03:48", "26/10/2015 03:47"),
    "2015-11-01": ("31/10/2015 03:56", "11/01/2016 03:47"),
}

def parse_joughin_tsx_time(file_path):
    # Read date string from filename
    match = re.search(r'TSX\.(\d{4}-\d{2}-\d{2})\.', os.path.basename(file_path))
    if not match:
        raise ValueError(f"Could not extract date pattern from {file_path}")
    
    date_key = match.group(1)  # e.g., '2009-03-31'
    
    # Find row in lookup table
    if date_key not in TSX_METADATA_TABLE:
        raise ValueError(f"Date '{date_key}' not found in metadata table.")
    
    # Read start date and end date from dictionary
    start_str, end_str = TSX_METADATA_TABLE[date_key]
    start_dt = datetime.strptime(start_str, "%d/%m/%Y %H:%M")
    end_dt = datetime.strptime(end_str, "%d/%m/%Y %H:%M")
    
    # Calculate mid_dt as midpoint
    mid_dt = start_dt + (end_dt - start_dt) / 2
    
    # Return outputs 
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_joughin_tsx(vx_path, master_x, master_y):
    vy_path = vx_path.replace('.vx.tif', '.vy.tif')
    
    # Load raw data
    vx_da = rioxarray.open_rasterio(vx_path).squeeze(drop=True)
    vy_da = rioxarray.open_rasterio(vy_path).squeeze(drop=True)
    
    # Calculate speed
    speed_raw = np.sqrt(vx_da**2 + vy_da**2)
    
    # Mask invalid data
    valid_mask = (vx_da > -200000) & (vx_da < 200000) & (vy_da > -200000) & (vy_da < 200000)
    speed_masked = speed_raw.where(valid_mask, np.nan)
    vx_masked = vx_da.where(valid_mask, np.nan)
    vy_masked = vy_da.where(valid_mask, np.nan)
    
    # Estimate errors as 5% of the absolute values
    speed_error_masked = speed_masked * 0.05
    vx_error_masked = np.abs(vx_masked) * 0.05
    vy_error_masked = np.abs(vy_masked) * 0.05
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["Joughin_TSX"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 10. Li Totten Processing ---
def parse_li_totten_time(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r'^(\d{4})(?:_(\d{4}))?_v\.tif$', filename)
    start_year = int(match.group(1))
    end_year = int(match.group(2)) if match.group(2) else start_year
    start_dt = datetime(start_year, 1, 1)
    end_dt = datetime(end_year, 12, 31)
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_li_totten(file_path, master_x, master_y):
    # Dynamically find the vx and vy file paths
    vx_path = file_path.replace('_v.tif', '_vx.tif')
    vy_path = file_path.replace('_v.tif', '_vy.tif')
    
    # Load raw data
    speed_raw = rioxarray.open_rasterio(file_path).squeeze(drop=True)
    vx_da = rioxarray.open_rasterio(vx_path).squeeze(drop=True)
    vy_da = rioxarray.open_rasterio(vy_path).squeeze(drop=True)
    
    # Force vx and vy to perfectly align with speed_raw's coordinates. 
    # This prevents AlignmentErrors caused by tiny floating-point differences in the source TIFFs.
    vx_da = vx_da.reindex_like(speed_raw, method='nearest')
    vy_da = vy_da.reindex_like(speed_raw, method='nearest')
    
    # Mask invalid data (ensure no nodata values slip through any of the arrays)
    valid_mask = (speed_raw > -200000) & (speed_raw < 200000) & \
                 (vx_da > -200000) & (vx_da < 200000) & \
                 (vy_da > -200000) & (vy_da < 200000)
                 
    speed_masked = speed_raw.where(valid_mask, np.nan)
    vx_masked = vx_da.where(valid_mask, np.nan)
    vy_masked = vy_da.where(valid_mask, np.nan)
    
    # Estimate errors as 5% of the absolute values
    speed_error_masked = speed_masked * 0.05
    vx_error_masked = np.abs(vx_masked) * 0.05
    vy_error_masked = np.abs(vy_masked) * 0.05
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["Li_Totten"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 11. ENVEO Sentinel-1 PIG Processing ---
def parse_enveo_s1_pig_time(file_path):
    match = re.search(r'_(\d{8})_(\d{8})_', os.path.basename(file_path))
    start_dt = datetime.strptime(match.group(1), "%Y%m%d")
    end_dt = datetime.strptime(match.group(2), "%Y%m%d")
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def preprocess_enveo_s1_pig(file_path, master_x, master_y):
    # Dynamically find the vxyz file path
    vxyz_path = file_path.replace('/mag/', '/vel/').replace('_vv.tif', '_vxyz.tif')
    
    # Load raw data
    speed_raw = rioxarray.open_rasterio(file_path).squeeze(drop=True)
    vxyz_da = rioxarray.open_rasterio(vxyz_path)
    
    # Extract vx (band 1) and vy (band 2)
    vx_raw = vxyz_da.sel(band=1).squeeze(drop=True)
    vy_raw = vxyz_da.sel(band=2).squeeze(drop=True)
    
    # Convert to m/year
    speed_yr = speed_raw * 365.25
    vx_yr = vx_raw * 365.25
    vy_yr = vy_raw * 365.25
    
    # Mask invalid data (using speed as the reference for valid pixels across all bands)
    valid_mask = (speed_yr >= 0) & (speed_yr <= 200000)
    speed_masked = speed_yr.where(valid_mask, np.nan)
    vx_masked = vx_yr.where(valid_mask, np.nan)
    vy_masked = vy_yr.where(valid_mask, np.nan)
    
    # Estimate errors as 5% of the absolute values
    speed_error_masked = speed_masked * 0.05
    vx_error_masked = np.abs(vx_masked) * 0.05
    vy_error_masked = np.abs(vy_masked) * 0.05
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["ENVEO_Sentinel-1_PIG"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 12. ENVEO ERS Processing ---
def catalog_enveo_ers(base_dir):
    epochs = []
    child_dirs = glob.glob(os.path.join(base_dir, "*_ERS_*"))
    for cdir in child_dirs:
        match = re.search(r's(\d{8})_e(\d{8})', os.path.basename(cdir))
        if not match: continue
        start_dt = datetime.strptime(match.group(1), "%Y%m%d")
        end_dt = datetime.strptime(match.group(2), "%Y%m%d")
        mid_dt = start_dt + (end_dt - start_dt) / 2
        t = np.datetime64(mid_dt)
        tb = np.array([np.datetime64(start_dt), np.datetime64(end_dt)])
        
        tifs = [t for t in glob.glob(os.path.join(cdir, "*.tif")) if "_mag" not in os.path.basename(t)]
        for tif in tifs:
            epochs.append({'time': t, 'time_bnds': tb, 'source': 'ENVEO_ERS', 'path': tif})
    return epochs

def preprocess_enveo_ers(file_path, master_x, master_y):
    # Load raw data
    da_raw = rioxarray.open_rasterio(file_path)
    vx_raw = da_raw.sel(band=1)
    vy_raw = da_raw.sel(band=2)
    
    # Convert components to m/year
    vx_yr = vx_raw * 365.25
    vy_yr = vy_raw * 365.25
    
    # Calculate speed in m/year
    speed_yr = np.sqrt(vx_yr**2 + vy_yr**2)
    
    # Mask invalid data
    valid_mask = (speed_yr >= 0) & (speed_yr <= 200000)
    speed_masked = speed_yr.where(valid_mask, np.nan)
    vx_masked = vx_yr.where(valid_mask, np.nan)
    vy_masked = vy_yr.where(valid_mask, np.nan)
    
    # Estimate errors as 5% of the absolute values
    speed_error_masked = speed_masked * 0.05
    vx_error_masked = np.abs(vx_masked) * 0.05
    vy_error_masked = np.abs(vy_masked) * 0.05
    
    # Drop band coordinate to avoid xarray dimension conflicts during interpolation
    speed_masked = speed_masked.drop_vars('band', errors='ignore')
    vx_masked = vx_masked.drop_vars('band', errors='ignore')
    vy_masked = vy_masked.drop_vars('band', errors='ignore')
    speed_error_masked = speed_error_masked.drop_vars('band', errors='ignore')
    vx_error_masked = vx_error_masked.drop_vars('band', errors='ignore')
    vy_error_masked = vy_error_masked.drop_vars('band', errors='ignore')
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["ENVEO_ERS"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 13. ENVEO TSX Processing ---
def catalog_enveo_tsx(base_dir):
    epochs = []
    child_dirs = glob.glob(os.path.join(base_dir, "*_TSX_*"))
    for cdir in child_dirs:
        tifs = [t for t in glob.glob(os.path.join(cdir, "*.tif")) if "mag" not in os.path.basename(t).lower()]
        for tif in tifs:
            match = re.search(r'_(\d{8})_(\d{8})_', os.path.basename(tif))
            if not match: continue
            start_dt = datetime.strptime(match.group(1), "%Y%m%d")
            end_dt = datetime.strptime(match.group(2), "%Y%m%d")
            mid_dt = start_dt + (end_dt - start_dt) / 2
            t = np.datetime64(mid_dt)
            tb = np.array([np.datetime64(start_dt), np.datetime64(end_dt)])
            epochs.append({'time': t, 'time_bnds': tb, 'source': 'ENVEO_TSX', 'path': tif})
    return epochs

def preprocess_enveo_tsx(file_path, master_x, master_y):
    # Load raw data
    da_raw = rioxarray.open_rasterio(file_path)
    vx_raw = da_raw.sel(band=1)
    vy_raw = da_raw.sel(band=2)
    
    # Convert components to m/year
    vx_yr = vx_raw * 365.25
    vy_yr = vy_raw * 365.25
    
    # Calculate speed in m/year
    speed_yr = np.sqrt(vx_yr**2 + vy_yr**2)
    
    # Mask invalid data
    valid_mask = (speed_yr >= 0) & (speed_yr <= 200000)
    speed_masked = speed_yr.where(valid_mask, np.nan)
    vx_masked = vx_yr.where(valid_mask, np.nan)
    vy_masked = vy_yr.where(valid_mask, np.nan)
    
    # Estimate errors as 5% of the absolute values
    speed_error_masked = speed_masked * 0.05
    vx_error_masked = np.abs(vx_masked) * 0.05
    vy_error_masked = np.abs(vy_masked) * 0.05
    
    # Drop band coordinate to avoid xarray dimension conflicts during interpolation
    speed_masked = speed_masked.drop_vars('band', errors='ignore')
    vx_masked = vx_masked.drop_vars('band', errors='ignore')
    vy_masked = vy_masked.drop_vars('band', errors='ignore')
    speed_error_masked = speed_error_masked.drop_vars('band', errors='ignore')
    vx_error_masked = vx_error_masked.drop_vars('band', errors='ignore')
    vy_error_masked = vy_error_masked.drop_vars('band', errors='ignore')
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["ENVEO_TSX"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 14. ENVEO ALOS Processing ---
def catalog_enveo_alos(base_dir):
    epochs = []
    child_dirs = glob.glob(os.path.join(base_dir, "*_ALOS_*"))
    for cdir in child_dirs:
        tifs = [t for t in glob.glob(os.path.join(cdir, "*.tif")) if "mag" not in os.path.basename(t).lower()]
        for tif in tifs:
            match = re.search(r'_(\d{8})_(\d{8})_', os.path.basename(tif))
            if not match: continue
            start_dt = datetime.strptime(match.group(1), "%Y%m%d")
            end_dt = datetime.strptime(match.group(2), "%Y%m%d")
            mid_dt = start_dt + (end_dt - start_dt) / 2
            t = np.datetime64(mid_dt)
            tb = np.array([np.datetime64(start_dt), np.datetime64(end_dt)])
            epochs.append({'time': t, 'time_bnds': tb, 'source': 'ENVEO_PALSAR', 'path': tif})
    return epochs

def preprocess_enveo_alos(file_path, master_x, master_y):
    # Load raw data
    da_raw = rioxarray.open_rasterio(file_path)
    vx_raw = da_raw.sel(band=1)
    vy_raw = da_raw.sel(band=2)
    
    # Convert components to m/year
    vx_yr = vx_raw * 365.25
    vy_yr = vy_raw * 365.25
    
    # Calculate speed in m/year
    speed_yr = np.sqrt(vx_yr**2 + vy_yr**2)
    
    # Mask invalid data
    valid_mask = (speed_yr >= 0) & (speed_yr <= 200000)
    speed_masked = speed_yr.where(valid_mask, np.nan)
    vx_masked = vx_yr.where(valid_mask, np.nan)
    vy_masked = vy_yr.where(valid_mask, np.nan)
    
    # Estimate errors as 5% of the absolute values
    speed_error_masked = speed_masked * 0.05
    vx_error_masked = np.abs(vx_masked) * 0.05
    vy_error_masked = np.abs(vy_masked) * 0.05
    
    # Drop band coordinate to avoid xarray dimension conflicts during interpolation
    speed_masked = speed_masked.drop_vars('band', errors='ignore')
    vx_masked = vx_masked.drop_vars('band', errors='ignore')
    vy_masked = vy_masked.drop_vars('band', errors='ignore')
    speed_error_masked = speed_error_masked.drop_vars('band', errors='ignore')
    vx_error_masked = vx_error_masked.drop_vars('band', errors='ignore')
    vy_error_masked = vy_error_masked.drop_vars('band', errors='ignore')
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["ENVEO_PALSAR"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 15. ENVEO TSX-S1 Processing ---
def catalog_enveo_tsx_s1(base_dir):
    epochs = []
    child_dirs = glob.glob(os.path.join(base_dir, "*_TSX-S1_*"))
    for cdir in child_dirs:
        tifs = glob.glob(os.path.join(cdir, "*_mag.tif"))
        for tif in tifs:
            start_dt = datetime(2015, 10, 31)
            end_dt = datetime(2016, 12, 11)
            mid_dt = start_dt + (end_dt - start_dt) / 2
            t = np.datetime64(mid_dt)
            tb = np.array([np.datetime64(start_dt), np.datetime64(end_dt)])
            epochs.append({'time': t, 'time_bnds': tb, 'source': 'ENVEO_TSX_Sentinel-1', 'path': tif})
    return epochs

def preprocess_enveo_tsx_s1(file_path, master_x, master_y):
    # Dynamically find the component file path by dropping "_mag"
    vxyz_path = file_path.replace('_mag.tif', '.tif')
    
    # Load raw data
    speed_raw = rioxarray.open_rasterio(file_path).squeeze(drop=True)
    vxyz_da = rioxarray.open_rasterio(vxyz_path)
    
    # Extract vx (band 1) and vy (band 2)
    vx_raw = vxyz_da.sel(band=1).squeeze(drop=True)
    vy_raw = vxyz_da.sel(band=2).squeeze(drop=True)
    
    # Convert to m/year
    speed_yr = speed_raw * 365.25
    vx_yr = vx_raw * 365.25
    vy_yr = vy_raw * 365.25
    
    # Mask invalid data
    valid_mask = (speed_yr > -200000) & (speed_yr < 200000)
    speed_masked = speed_yr.where(valid_mask, np.nan)
    vx_masked = vx_yr.where(valid_mask, np.nan)
    vy_masked = vy_yr.where(valid_mask, np.nan)
    
    # Estimate errors as 5% of the absolute values
    speed_error_masked = speed_masked * 0.05
    vx_error_masked = np.abs(vx_masked) * 0.05
    vy_error_masked = np.abs(vy_masked) * 0.05
    
    # Drop band coordinate to avoid xarray dimension conflicts during interpolation
    speed_masked = speed_masked.drop_vars('band', errors='ignore')
    vx_masked = vx_masked.drop_vars('band', errors='ignore')
    vy_masked = vy_masked.drop_vars('band', errors='ignore')
    speed_error_masked = speed_error_masked.drop_vars('band', errors='ignore')
    vx_error_masked = vx_error_masked.drop_vars('band', errors='ignore')
    vy_error_masked = vy_error_masked.drop_vars('band', errors='ignore')
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["ENVEO_TSX_Sentinel-1"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds
# --- 16. ENVEO TSX-PALSAR Processing ---
def catalog_enveo_tsx_palsar(base_dir):
    epochs = []
    child_dirs = glob.glob(os.path.join(base_dir, "*_TSX-PALSAR_*"))
    for cdir in child_dirs:
        tifs = glob.glob(os.path.join(cdir, "*_mag.tif"))
        for tif in tifs:
            start_dt = datetime(2010, 7, 2)
            end_dt = datetime(2012, 3, 9)
            mid_dt = start_dt + (end_dt - start_dt) / 2
            t = np.datetime64(mid_dt)
            tb = np.array([np.datetime64(start_dt), np.datetime64(end_dt)])
            epochs.append({'time': t, 'time_bnds': tb, 'source': 'ENVEO_TSX_PALSAR', 'path': tif})
    return epochs

def preprocess_enveo_tsx_palsar(file_path, master_x, master_y):
    # Dynamically find the component file path by dropping "_mag"
    vxyz_path = file_path.replace('_mag.tif', '.tif')
    
    # Load raw data
    speed_raw = rioxarray.open_rasterio(file_path).squeeze(drop=True)
    vxyz_da = rioxarray.open_rasterio(vxyz_path)
    
    # Extract vx (band 1) and vy (band 2)
    vx_raw = vxyz_da.sel(band=1).squeeze(drop=True)
    vy_raw = vxyz_da.sel(band=2).squeeze(drop=True)
    
    # Convert to m/year
    speed_yr = speed_raw * 365.25
    vx_yr = vx_raw * 365.25
    vy_yr = vy_raw * 365.25
    
    # Mask invalid data
    valid_mask = (speed_yr > -200000) & (speed_yr < 200000)
    speed_masked = speed_yr.where(valid_mask, np.nan)
    vx_masked = vx_yr.where(valid_mask, np.nan)
    vy_masked = vy_yr.where(valid_mask, np.nan)
    
    # Estimate errors as 5% of the absolute values
    speed_error_masked = speed_masked * 0.05
    vx_error_masked = np.abs(vx_masked) * 0.05
    vy_error_masked = np.abs(vy_masked) * 0.05
    
    # Drop band coordinate to avoid xarray dimension conflicts during interpolation
    speed_masked = speed_masked.drop_vars('band', errors='ignore')
    vx_masked = vx_masked.drop_vars('band', errors='ignore')
    vy_masked = vy_masked.drop_vars('band', errors='ignore')
    speed_error_masked = speed_error_masked.drop_vars('band', errors='ignore')
    vx_error_masked = vx_error_masked.drop_vars('band', errors='ignore')
    vy_error_masked = vy_error_masked.drop_vars('band', errors='ignore')
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array(["ENVEO_TSX_PALSAR"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 17. SHIFT Processing ---
def parse_shift_time(date_str):
    start_str, end_str = date_str.split('_')
    start_dt = datetime.strptime(start_str, "%Y%m%d")
    end_dt = datetime.strptime(end_str, "%Y%m%d")
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def read_shift_med(filepath):
    if not os.path.exists(filepath):
        return np.nan
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) < 2: return np.nan
        headers = [h.lower() for h in lines[0].split()]
        values = lines[1].split()
        med_idx = -1
        if 'med' in headers:
            med_idx = headers.index('med')
        elif len(values) >= 6:
            med_idx = 5
        if med_idx != -1 and med_idx < len(values):
            val_str = values[med_idx].lower()
            if val_str in ['nan', 'none', 'null', 'n/a', '-9999', '-1', '-9999.0', '-1.0']:
                return np.nan
            val = float(values[med_idx])
            if np.isnan(val) or val == -9999 or val == -1:
                return np.nan
            return val
    except Exception as e:
        print(f"Warning: Failed to parse error from {filepath}: {e}")
        pass
    return np.nan

def read_shift_std(filepath):
    """Safely reads the 'std' column from a SHIFT metadata text file."""
    if not os.path.exists(filepath):
        return np.nan
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2: 
            return np.nan
            
        headers = lines[0].split()
        values = lines[1].split()
        
        if 'std' in headers:
            std_idx = headers.index('std')
            val = float(values[std_idx])
            # Return NaN if the parsed value is basically nodata
            if np.isnan(val) or val == -9999 or val == -1:
                return np.nan
            return val
    except:
        pass
    return np.nan

def catalog_shift(base_dir):
    epochs = []
    if not os.path.exists(base_dir):
        print(f"Warning: SHIFT base directory not found: {base_dir}")
        return epochs
    for d in os.listdir(base_dir):
        if re.match(r'^\d{8}_\d{8}$', d):
            cdir = os.path.join(base_dir, d)
            if not os.path.isdir(cdir): continue
            date_str = d
            try:
                t, tb = parse_shift_time(date_str)
            except ValueError: continue
            speed_file = os.path.join(cdir, f"S_{date_str}_200m_raw.tif")
            if not os.path.exists(speed_file):
                tifs = [f for f in os.listdir(cdir) if f.startswith('S_') and 'raw.tif' in f.lower()]
                if tifs: speed_file = os.path.join(cdir, tifs[0])
                else: continue
            epochs.append({
                'time': t, 'time_bnds': tb, 
                'source': 'SHIFT', 'path': speed_file,
                'epoch_dir': cdir, 'date_str': date_str
            })
    return epochs

def preprocess_shift(file_path, epoch_dir, date_str, master_x, master_y):
    meta_dir = os.path.join(epoch_dir, "metadata")
    
    # Safely determine paths for U (vx) and V (vy) files
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    u_path = os.path.join(dir_name, base_name.replace(f"S_{date_str}", f"U_{date_str}"))
    v_path = os.path.join(dir_name, base_name.replace(f"S_{date_str}", f"V_{date_str}"))
    
    # Load raw data
    speed = rioxarray.open_rasterio(file_path).squeeze('band', drop=True)
    vx = rioxarray.open_rasterio(u_path).squeeze('band', drop=True)
    vy = rioxarray.open_rasterio(v_path).squeeze('band', drop=True)
    
    # Create and apply valid mask
    valid_mask = (speed > -200000) & (speed < 200000)
    if speed.rio.nodata is not None: 
        valid_mask = valid_mask & (speed != speed.rio.nodata)
        
    speed_masked = speed.where(valid_mask, np.nan)
    vx_masked = vx.where(valid_mask, np.nan)
    vy_masked = vy.where(valid_mask, np.nan)
    
    # Interpolate the masked data
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Determine error scalars from metadata
    rock_u = read_shift_std(os.path.join(meta_dir, f"rock_mask_U_metadata_200m_{date_str}.txt"))
    rock_v = read_shift_std(os.path.join(meta_dir, f"rock_mask_V_metadata_200m_{date_str}.txt"))
    
    err_u_scalar = np.nan
    err_v_scalar = np.nan
    
    # 1. Prefer rock_u / rock_v
    if not (np.isnan(rock_u) or np.isnan(rock_v)):
        err_u_scalar = rock_u
        err_v_scalar = rock_v
    else:
        # 2. Fall back to off_u / off_v
        off_u = read_shift_std(os.path.join(meta_dir, f"off_ice_U_metadata_200m_{date_str}.txt"))
        off_v = read_shift_std(os.path.join(meta_dir, f"off_ice_V_metadata_200m_{date_str}.txt"))
        if not (np.isnan(off_u) or np.isnan(off_v)):
            err_u_scalar = off_u
            err_v_scalar = off_v
            
    # Apply errors (either scalars across the domain or 5% of velocity values)
    if not np.isnan(err_u_scalar):
        # Create full arrays matching the valid data footprint
        vx_error_interp = xr.full_like(vx_interp, err_u_scalar).where(vx_interp.notnull())
        vy_error_interp = xr.full_like(vy_interp, err_v_scalar).where(vy_interp.notnull())
        
        # Calculate speed scalar error from component scalar errors
        speed_err_scalar = np.sqrt(err_u_scalar**2 + err_v_scalar**2)
        speed_error_interp = xr.full_like(speed_interp, speed_err_scalar).where(speed_interp.notnull())
    else:
        # 3. Fall back to 5% of absolute values
        vx_error_interp = np.abs(vx_interp) * 0.05
        vy_error_interp = np.abs(vy_interp) * 0.05
        speed_error_interp = speed_interp * 0.05
        
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year', 'grid_mapping': 'spatial_ref'}),
        'data_source': (['time'], np.array(["SHIFT"], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- 18. Historical Processing ---
HISTORICAL_METADATA = [
    {"file": "PIGLS74.vx.tif", "date1": "24/01/1973", "date2": "13/02/1975", "source": "Rosanova_PIG74"}, # Rosanova, C. E., and B. K. Lucchitta, Acceleration of Pine Island Glacier Ice Shelf, Eos Trans. AGU, 82(47), 2001
    {"file": "PIGLS87.vx.tif", "date1": "21/12/1986", "date2": "18/12/1988", "source": "Scambos_PIG87"}, # Scambos, T., B. Raup, and J. Bohlander, compilers, Antarctic ice velocity data archive. Boulder, CO: National Snow and Ice Data Center. Digital media (http://nsidc.org/data/velmap/), 2003
    {"file": "PIG92.vx.tif", "date1": "29/02/1992", "date2": "06/03/1992", "source": "Joughin_PIG_ERS"}, # Joughin, I., Rignot, E., Rosanova, C.E., Lucchitta, B.K. and Bohlander, J., 2003. Timing of recent accelerations of Pine Island glacier, Antarctica. Geophysical Research Letters, 30(13).
    {"file": "PIG94.vx.tif", "date1": "22/01/1994", "date2": "25/01/1994", "source": "Joughin_PIG_ERS"}, 
    {"file": "PIG96.vx.tif", "date1": "21/10/1995", "date2": "25/04/1996", "source": "Joughin_PIG_ERS"}, 
    {"file": "PIG00.vx.tif", "date1": "20/11/1999", "date2": "04/03/2000", "source": "Joughin_PIG_ERS"},
    {"file": "sulzberger_vx.tif", "date1": "11/12/1986", "date2": "16/03/1989", "source": "Bohlander_Scambos02"}, # Bohlander, J., & Scambos, T. A. (2002). Ice Flow and Morphology of Prestrud Inlet and Western Sulzberger Ice Shelf, West Antarctica. Polar Geography, 26(3), 227–234. https://doi.org/10.1080/789610194
    {"file": "prestud_vx.tif", "date1": "14/01/1987", "date2": "16/03/1989", "source": "Bohlander_Scambos02"},
    {"file": "streamd1_vx.tif", "date1": "15/01/1987", "date2": "12/12/1991", "source": "Bindschadler96"}, # Bindschadler, R., Vornberger, P., Blankenship, D., Scambos, T. and Jacobel, R., 1996. Surface velocity and mass balance of Ice Streams D and E, West Antarctica. Journal of Glaciology, 42(142), pp.461-475.
    {"file": "streamd2_vx.tif", "date1": "24/01/1985", "date2": "12/12/1991", "source": "Bindschadler96"},
    {"file": "streamd3_vx.tif", "date1": "16/01/1987", "date2": "12/12/1988", "source": "Bindschadler96"},
    {"file": "streamde_vx.tif", "date1": "16/01/1987", "date2": "12/12/1988", "source": "Bindschadler96"},
    {"file": "lowde6_vx.tif", "date1": "15/01/1987", "date2": "05/02/1990", "source": "Bindschadler96"},
    {"file": "lowde7_vx.tif", "date1": "15/02/1989", "date2": "05/02/1990", "source": "Bindschadler96"},
    {"file": "lowde8_vx.tif", "date1": "15/01/1987", "date2": "15/02/1989", "source": "Bindschadler96"},
    {"file": "streamdsp_vx.tif", "date1": "20/01/1994", "date2": "09/12/1995", "source": "ScambosBohlander_DF"}, # only reference provided is: Scambos, T.A., and J.A. Bohlander. "Ice velocity mapping of Ice Streams D and F: flow dynamics and changes. (in prep)
    {"file": "streamf_vx.tif", "date1": "14/01/1987", "date2": "05/02/1990", "source": "ScambosBohlander_DF"},   # so let's use the only place this data can be downloaded from instead: Bindschadler, R., & Raymond, C. (2004) "Antarctic Ice Velocity Data" U.S. Antarctic Program (USAP) Data Center. doi: https://doi.org/10.7265/N5KS6PH5.
    {"file": "institute_vx.tif", "date1": "15/02/1986", "date2": "05/01/1989", "source": "Scambos04"}, # Scambos, T., Bohlander, J., Raup, B. and Haran, T., 2004. Glaciological characteristics of Institute Ice Stream using remote sensing. Antarctic Science, 16(2), pp.205-213.
    {"file": "ByrdGlacier_1978-1979_velmap_vx.tif", "date1": "10/12/1978", "date2": "06/02/1979", "source": "Brecher82"}, # Brecher, H.H., 1982. Photographic determination of surface velocities and elevations on Byrd Glacier. Antarctic Journal of the United States, 17(5), pp.79-81. (( 56 days apart during the 77/79 Antarctic summer; the actual dates are given in this pdf: https://s3.amazonaws.com/Antarctica/AJUS/AJUSvXIVn5/AJUSvXIVn5p88.pdf))
    {"file": "filch_ronne_vx.tif", "date1": "16/02/1986", "date2": "01/01/1994", "source": "Vaughan96"} # Vaughan, D.G., and M. Jonas, 1996. Measurements of velocity of Filchner-Ronne Ice Shelf. Filchner-Ronne Ice Shelf Programme (FRISP) Report No. 10. Ed. H Oerter. Pub. AWI, Bremerhaven. 111-116. Date1 is given in Swithinbank et al 1988: https://www.cambridge.org/core/journals/annals-of-glaciology/article/glaciological-map-of-filchnerronne-ice-shelf-antarctica/28B933A9FD82A24F54E4B6E0FBC95737. No dates for date2 are available, only the year. 
]

def parse_historical_time(date1_str, date2_str):
    """Parses time bounds from the DD/MM/YYYY format."""
    start_dt = datetime.strptime(date1_str, "%d/%m/%Y")
    end_dt = datetime.strptime(date2_str, "%d/%m/%Y")
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return np.datetime64(mid_dt), np.array([np.datetime64(start_dt), np.datetime64(end_dt)])

def find_historical_error_paths(vx_path, vy_path):
    """
    Searches for matching error geotiffs using possible naming variations
    ('err_vx'/'err_vy' or 'errvx'/'errvy', preserving '_' or '.' delimiters).
    """
    candidate_patterns = [
        # Direct replacement of vx -> err_vx / errvx
        (re.sub(r'vx(\.tif)$', r'err_vx\1', vx_path, flags=re.IGNORECASE),
         re.sub(r'vy(\.tif)$', r'err_vy\1', vy_path, flags=re.IGNORECASE)),
        (re.sub(r'vx(\.tif)$', r'errvx\1', vx_path, flags=re.IGNORECASE),
         re.sub(r'vy(\.tif)$', r'errvy\1', vy_path, flags=re.IGNORECASE)),
        # Delimiter-aware replacements
        (re.sub(r'([._])vx(\.tif)$', r'\1err_vx\2', vx_path, flags=re.IGNORECASE),
         re.sub(r'([._])vy(\.tif)$', r'\1err_vy\2', vy_path, flags=re.IGNORECASE)),
        (re.sub(r'([._])vx(\.tif)$', r'\1errvx\2', vx_path, flags=re.IGNORECASE),
         re.sub(r'([._])vy(\.tif)$', r'\1errvy\2', vy_path, flags=re.IGNORECASE)),
    ]
    
    for err_vx, err_vy in candidate_patterns:
        if os.path.exists(err_vx) and os.path.exists(err_vy):
            return err_vx, err_vy

    return None, None


def preprocess_historical(file_path, source_label, master_x, master_y):
    vx_path = file_path
    # Dynamically match either .vx.tif or _vx.tif and convert to vy equivalent
    vy_path = re.sub(r'vx(\.tif)$', r'vy\1', vx_path, flags=re.IGNORECASE)
    
    # Load raw velocity data
    vx_da = rioxarray.open_rasterio(vx_path).squeeze(drop=True)
    vy_da = rioxarray.open_rasterio(vy_path).squeeze(drop=True)
    
    # Align vy to vx to ensure coordinates perfectly match before math
    vy_da = vy_da.reindex_like(vx_da, method='nearest')
    
    # Calculate speed (magnitude)
    speed_raw = np.sqrt(vx_da**2 + vy_da**2)
    
    # Mask invalid data
    valid_mask = (vx_da > -200000) & (vx_da < 200000) & \
                 (vy_da > -200000) & (vy_da < 200000)
                 
    speed_masked = speed_raw.where(valid_mask, np.nan)
    vx_masked = vx_da.where(valid_mask, np.nan)
    vy_masked = vy_da.where(valid_mask, np.nan)
    
    # Check for error geotiffs
    err_vx_path, err_vy_path = find_historical_error_paths(vx_path, vy_path)

    if err_vx_path and err_vy_path:
        err_vx_da = rioxarray.open_rasterio(err_vx_path).squeeze(drop=True).reindex_like(vx_da, method='nearest')
        err_vy_da = rioxarray.open_rasterio(err_vy_path).squeeze(drop=True).reindex_like(vx_da, method='nearest')

        vx_error_masked = err_vx_da.where(valid_mask, np.nan)
        vy_error_masked = err_vy_da.where(valid_mask, np.nan)
        
        # Quadrature combination for speed error magnitude
        speed_error_masked = np.sqrt(vx_error_masked**2 + vy_error_masked**2)
    else:
        # Fallback estimation (10%) if error rasters are not found on disk
        speed_error_masked = speed_masked * 0.10
        vx_error_masked = np.abs(vx_masked) * 0.10
        vy_error_masked = np.abs(vy_masked) * 0.10
    
    # Interpolate to master grid
    speed_interp = speed_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_interp = vx_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_interp = vy_masked.interp(x=master_x, y=master_y, method="nearest")
    speed_error_interp = speed_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vx_error_interp = vx_error_masked.interp(x=master_x, y=master_y, method="nearest")
    vy_error_interp = vy_error_masked.interp(x=master_x, y=master_y, method="nearest")
    
    # Extract values and round
    speed_vals = np.round(speed_interp.values, 1)
    vx_vals = np.round(vx_interp.values, 1)
    vy_vals = np.round(vy_interp.values, 1)
    speed_error_vals = np.round(speed_error_interp.values, 1)
    vx_error_vals = np.round(vx_error_interp.values, 1)
    vy_error_vals = np.round(vy_error_interp.values, 1)
    
    # Package into standardized Dataset using the dynamic source_label
    ds = xr.Dataset({
        'speed': (['time', 'y', 'x'], np.expand_dims(speed_vals, axis=0), {'units': 'm/year'}),
        'vx': (['time', 'y', 'x'], np.expand_dims(vx_vals, axis=0), {'units': 'm/year'}),
        'vy': (['time', 'y', 'x'], np.expand_dims(vy_vals, axis=0), {'units': 'm/year'}),
        'speed_error': (['time', 'y', 'x'], np.expand_dims(speed_error_vals, axis=0), {'units': 'm/year'}),
        'vx_error': (['time', 'y', 'x'], np.expand_dims(vx_error_vals, axis=0), {'units': 'm/year'}),
        'vy_error': (['time', 'y', 'x'], np.expand_dims(vy_error_vals, axis=0), {'units': 'm/year'}),
        'data_source': (['time'], np.array([source_label], dtype="<U50"))
    }, coords={'y': (['y'], master_y.values), 'x': (['x'], master_x.values)})
    
    return ds

# --- HPC BUILDER LOGIC ---

def build_catalog_and_skeleton():
    """Run ONCE by a single job to map files and create the Zarr structure."""
    print("Cataloging files and generating noisy unique times...", flush=True)
    epochs = []
    
    # 1. ENVEO
    for f in sorted(glob.glob(os.path.join(ENVEO_DIR, "*.nc"))):
        t, tb = parse_enveo_monthly_time(f)
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'ENVEO_monthly', 'path': f})
        
    # 2. ITS_LIVE
    for f in sorted(glob.glob(os.path.join(ITSLIVE_DIR, "*.nc"))):
        t, tb = parse_itslive_annual_time(f)
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'ITS_LIVE_annual', 'path': f})

    # 3. MEaSUREs Annual
    for f in sorted(glob.glob(os.path.join(MEASURES_DIR, "*_1km_v*.nc"))):
        t, tb = parse_measures_annual_time(f)
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'MEaSUREs_annual', 'path': f})

    # 4. MEaSUREs Multiyear
    for f in sorted(glob.glob(os.path.join(MEASURES_MULTI_DIR, "*.nc"))):
        t, tb = parse_measures_multiyear_time(f)
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'MEaSUREs_multiyear', 'path': f})
        
    # 5. MEaSUREs ASE
    for f in sorted(glob.glob(os.path.join(MEASURES_ASE_DIR, "*.nc"))):
        epochs.extend(catalog_measures_ase(f))
        
    # 6. SID Annual
    for f in sorted(glob.glob(os.path.join(SID_ANNUAL_DIR, "*.nc"))):
        t, tb = parse_sid_annual_time(f)
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'SID_annual', 'path': f})
        
    # 7. ESA CCI Annual
    for f in sorted(glob.glob(os.path.join(ESA_CCI_ANNUAL_DIR, "*.nc"))):
        t, tb = parse_esacci_annual_time(f)
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'C3S_annual', 'path': f})

    # 8. Joughin Sentinel-1
    for f in sorted(glob.glob(os.path.join(JOUGHIN_DIR, "VelPIG.S1Quarterly.*.vx.tif"))):
        t, tb = parse_joughin_s1_time(f)
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'Joughin_Sentinel-1', 'path': f})

    # 9. Joughin TSX
    for f in sorted(glob.glob(os.path.join(JOUGHIN_DIR, "VelPIG.TSX.*.vx.tif"))):
        t, tb = parse_joughin_tsx_time(f)
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'Joughin_TSX', 'path': f})

    # 10. Li Totten
    li_totten_targets = ["1963_1973_v.tif", "1973_1989_v.tif", "1989_v.tif"]
    for filename in li_totten_targets:
        file_path = os.path.join(LI_TOTTEN_DIR, filename)
        if os.path.exists(file_path):
            t, tb = parse_li_totten_time(file_path)
            epochs.append({'time': t, 'time_bnds': tb, 'source': 'Li_Totten', 'path': file_path})
            
    # 11. ENVEO Sentinel-1 PIG
    for f in sorted(glob.glob(os.path.join(ENVEO_S1_PIG_DIR, "*_vv.tif"))):
        t, tb = parse_enveo_s1_pig_time(f)
        epochs.append({'time': t, 'time_bnds': tb, 'source': 'ENVEO_Sentinel-1_PIG', 'path': f})
        
    # 12. ENVEO ERS
    epochs.extend(catalog_enveo_ers(ENVEO_OTHERS_DIR))
    
    # 13. ENVEO TSX
    epochs.extend(catalog_enveo_tsx(ENVEO_OTHERS_DIR))
    
    # 14. ENVEO ALOS
    epochs.extend(catalog_enveo_alos(ENVEO_OTHERS_DIR))
    
    # 15. ENVEO TSX-S1
    epochs.extend(catalog_enveo_tsx_s1(ENVEO_OTHERS_DIR))
    
    # 16. ENVEO TSX-PALSAR
    epochs.extend(catalog_enveo_tsx_palsar(ENVEO_OTHERS_DIR))
    
    # 17. SHIFT
    epochs.extend(catalog_shift(SHIFT_DIR))
    
    # 18. Historical data
    for meta in HISTORICAL_METADATA:
        historical_file = os.path.join(HISTORICAL_DIR, meta["file"])
        if os.path.exists(historical_file):
            t, tb = parse_historical_time(meta["date1"], meta["date2"])
            epochs.append({'time': t, 'time_bnds': tb, 'source': meta["source"], 'path': historical_file})
        
    # Sort ALL chronologically by the true base time
    epochs = sorted(epochs, key=lambda d: d['time'])
    
    # Guarantee strict uniqueness deterministically
    for i in range(1, len(epochs)):
        if epochs[i]['time'] <= epochs[i-1]['time']:
            # Calculate the exact offset needed to be 1ms greater than the previous time
            diff = epochs[i-1]['time'] - epochs[i]['time']
            offset = diff + np.timedelta64(1, 'ms')
            
            # Apply to both time and time_bnds
            epochs[i]['time'] += offset
            epochs[i]['time_bnds'] += offset

    total_epochs = len(epochs)
    print(f"Found {total_epochs} total epochs for the FULL build.")

    # Save to disk so workers know the master index map
    with open(CATALOG_FILE, 'wb') as f:
        pickle.dump(epochs, f)

    print("Extracting master grid from the first ENVEO file...", flush=True)
    first_enveo = next(item['path'] for item in epochs if item['source'] == 'ENVEO_monthly')
    ds_master = preprocess_enveo_monthly(first_enveo)
    master_x, master_y = ds_master['x'], ds_master['y']
    
    # Extract times, time bounds and sources from the catalog
    times = [ep['time'] for ep in epochs]
    time_bnds = [ep['time_bnds'] for ep in epochs]
    sources = np.array([ep['source'] for ep in epochs], dtype="<U50")
    
    # Chunk size for time is 1 to allow parallel worker writes
    chunk_def = (1, 1024, 1024) 
    
    # Create empty 3D dask array
    empty_array_3D = da.empty((total_epochs, master_y.size, master_x.size), chunks=chunk_def, dtype=np.float32)
    
    ds_skeleton = xr.Dataset({
        'speed': (['time', 'y', 'x'], empty_array_3D, ds_master['speed'].attrs),
        'vx': (['time', 'y', 'x'], empty_array_3D, ds_master['vx'].attrs),
        'vy': (['time', 'y', 'x'], empty_array_3D, ds_master['vy'].attrs),
        'speed_error': (['time', 'y', 'x'], empty_array_3D, ds_master['speed_error'].attrs),
        'vx_error': (['time', 'y', 'x'], empty_array_3D, ds_master['vx_error'].attrs),
        'vy_error': (['time', 'y', 'x'], empty_array_3D, ds_master['vy_error'].attrs),
        'spatial_ref': ([], ds_master['spatial_ref'].values, ds_master['spatial_ref'].attrs),
        'data_source': (['time'], sources)
    }, coords={
        'time': (['time'], np.array(times)),
        'time_bnds': (['time', 'bnds'], np.array(time_bnds)),
        'y': (['y'], master_y.values, ds_master['y'].attrs),
        'x': (['x'], master_x.values, ds_master['x'].attrs),
    })
        
    ds_skeleton.attrs.update(ds_master.attrs)
    ds_skeleton.attrs['dataset_version'] = TARGET_VERSION
    ds_skeleton.attrs['last_updated'] = date.today().isoformat()

    for var in ['time', 'time_bnds']: ds_skeleton[var].encoding.clear()
    encoding = {
            'speed': {'chunks': chunk_def}, 
            'vx': {'chunks': chunk_def}, 
            'vy': {'chunks': chunk_def}, 
            'speed_error': {'chunks': chunk_def}, 
            'vx_error': {'chunks': chunk_def}, 
            'vy_error': {'chunks': chunk_def}, 
            'data_source': {'chunks': (total_epochs,), 'dtype': '<U50'} 
        }
    
    print("Writing skeleton to disk...", flush=True)
    ds_skeleton.to_zarr(OUTPUT_ZARR, compute=False, encoding=encoding, mode='w')
    print("Skeleton initialized successfully!", flush=True)


def process_worker(target_source, batch_start=None, batch_end=None):
    """Run by multiple parallel jobs. Loads catalog, filters by source, and inserts data in batches."""
    print(f"Worker started for source: {target_source}", flush=True)
    with open(CATALOG_FILE, 'rb') as f:
        epochs = pickle.load(f)
        
    # We need the master grid for interpolation
    first_enveo = next(item['path'] for item in epochs if item['source'] == 'ENVEO_monthly')
    
    # Use the master dataset logic (assuming preprocess_enveo_monthly can run with just the file path initially)
    ds_master = preprocess_enveo_monthly(first_enveo)
    master_x, master_y = ds_master['x'], ds_master['y']
    
    # 1. Filter for source while retaining the global index 'i' for the Zarr region write
    source_epochs = [(i, ep) for i, ep in enumerate(epochs) if ep['source'] == target_source]
    
    # 2. Slice the list if batch arguments were provided
    if batch_start is not None and batch_end is not None:
        source_epochs = source_epochs[batch_start:batch_end]
        
    print(f"Processing {len(source_epochs)} epochs for {target_source} (Indices {batch_start} to {batch_end})", flush=True)
    
    for global_i, ep in source_epochs:
        display_name = f"{os.path.basename(ep['path'])} ({ep.get('year', '')})" if 'year' in ep else os.path.basename(ep['path'])
        print(f"  [Global Index: {global_i}] Processing {ep['source']}: {display_name}", flush=True)
        
        # Dispatch to appropriate function
        if ep['source'] == 'ENVEO_monthly': ds_slice = preprocess_enveo_monthly(ep['path'], master_x, master_y)
        elif ep['source'] == 'ITS_LIVE_annual': ds_slice = preprocess_itslive_annual(ep['path'], master_x, master_y)
        elif ep['source'] == 'MEaSUREs_annual': ds_slice = preprocess_measures_annual(ep['path'], master_x, master_y)
        elif ep['source'] == 'MEaSUREs_multiyear': ds_slice = preprocess_measures_multiyear(ep['path'], master_x, master_y)
        elif ep['source'] == 'MEaSUREs_ASE': ds_slice = preprocess_measures_ase(ep['path'], ep['year'], master_x, master_y)
        elif ep['source'] == 'SID_annual': ds_slice = preprocess_sid_annual(ep['path'], master_x, master_y)
        elif ep['source'] == 'C3S_annual': ds_slice = preprocess_esacci_annual(ep['path'], master_x, master_y)
        elif ep['source'] == 'Joughin_Sentinel-1': ds_slice = preprocess_joughin_s1(ep['path'], master_x, master_y)
        elif ep['source'] == 'Joughin_TSX': ds_slice = preprocess_joughin_tsx(ep['path'], master_x, master_y)
        elif ep['source'] == 'Li_Totten': ds_slice = preprocess_li_totten(ep['path'], master_x, master_y)
        elif ep['source'] == 'ENVEO_Sentinel-1_PIG': ds_slice = preprocess_enveo_s1_pig(ep['path'], master_x, master_y)
        elif ep['source'] == 'ENVEO_ERS': ds_slice = preprocess_enveo_ers(ep['path'], master_x, master_y)
        elif ep['source'] == 'ENVEO_TSX': ds_slice = preprocess_enveo_tsx(ep['path'], master_x, master_y)
        elif ep['source'] == 'ENVEO_PALSAR': ds_slice = preprocess_enveo_alos(ep['path'], master_x, master_y)
        elif ep['source'] == 'ENVEO_TSX_Sentinel-1': ds_slice = preprocess_enveo_tsx_s1(ep['path'], master_x, master_y)
        elif ep['source'] == 'ENVEO_TSX_PALSAR': ds_slice = preprocess_enveo_tsx_palsar(ep['path'], master_x, master_y)
        elif ep['source'] == 'SHIFT': ds_slice = preprocess_shift(ep['path'], ep['epoch_dir'], ep['date_str'], master_x, master_y)
        elif ep['source'] in ['Rosanova_PIG74', 'Scambos_PIG87', 'Joughin_PIG_ERS', 'Bohlander_Scambos02', 'Bindschadler96', 'ScambosBohlander_DF', 'Scambos04', 'Brecher82', 'Vaughan96']: ds_slice = preprocess_historical(ep['path'], ep['source'], master_x, master_y)
            
        # 3. Drop structural/spatial/time coordinates before mapping payload onto the skeleton store
        ds_to_write = ds_slice.drop_vars(['x', 'y', 'spatial_ref', 'time', 'time_bnds', 'source'], errors='ignore')
        
        # 4. Use the global index to write to the exact correct time coordinate slice
        ds_to_write.to_zarr(OUTPUT_ZARR, region={'time': slice(global_i, global_i+1)})
        
    print(f"Worker for {target_source} (Indices {batch_start}-{batch_end}) completed!", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [init | process SOURCE_NAME | consolidate]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "init":
        build_catalog_and_skeleton()
    elif command == "process":
        if len(sys.argv) < 3:
            print("Please provide a data source name. (e.g. python script.py process ENVEO_monthly)")
            sys.exit(1)
            
        source_target = sys.argv[2]
        batch_start, batch_end = None, None
        
        # If batch boundaries are provided via CLI arguments
        if len(sys.argv) >= 5:
            batch_start = int(sys.argv[3])
            batch_end = int(sys.argv[4])
            
        process_worker(source_target, batch_start, batch_end)
    elif command == "consolidate":
        zarr.consolidate_metadata(OUTPUT_ZARR)
        print("Zarr metadata consolidated successfully!", flush=True)