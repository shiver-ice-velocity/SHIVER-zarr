# The Sheffield Ice Velocity ExploreR (SHIVER): A unified satellite-derived ice velocity dataset for Earth's ice sheets

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21375859.svg)](https://doi.org/10.5281/zenodo.21375859)

**NOTE:** This code and the associated manuscript are in preparation and may be subject to change.

For more details, check out the [**SHIVER readthedocs**](https://shiver-zarr.readthedocs.io/)

## Overview
This repository contains:
1. The scripts used to generate the dataset described in the manuscript in preparation: *"The Sheffield Ice Velocity ExploreR (SHIVER): an online tool for low latency exploration, analysis and sub-setting of unified satellite-derived ice velocity data for Earth's ice sheets"*. 
2. Everything required to produce the SHIVER readthedocs website (https://shiver-zarr.readthedocs.io/), which describes how to access cloud-hosted Zarr stores of the SHIVER data.

**Note:** This repository only contains the code to generate the dataset. The resulting dataset is hosted by Source Cooperative at https://source.coop/uos-shiver.

## Directory Structure
* `src/` : Contains the Python scripts used to generate the unified dataset.
	* `citations.py` : Provides formatted citation records for each contributing dataset. 
	* `greenland/` 
		* `build_multisource_zarr.py` : Compiles a Zarr store (*spatial_200.zarr) from all contributing datasets for Greenland. The chunk definition is 1024 x 1024 x 1 (x, y, t) and the resolution is 200 x 200 m.
		* `build_multisource_omezarr.py` : Compiles an OME-Zarr store (*spatial.zarr) from all contributing datasets for Greenland. The chunk definition is 1024 x 1024 x 1 (x, y, t) and the resolution is telescopes from 200 m to 3200 m.
		* `rechunk_zarr_safe.py` : Rechunks the *spatial_200.zarr Zarr store to either a time-series optimised (chunk def: 64 x 64 x -1) Zarr or general-purpose Zarr (chunk def: 256 x 256 x 300). 
		* `update/` 
			* `scrape_promice.py` : Download only new PROMICE (edition 5) files.
			* `append_new_promice_to_spatial_200_zarr.py` : Appends new PROMICE files to the spatial_200 Zarr store.
			* `append_new_promice_to_spatial_zarr.py` : Appends new PROMICE files to the spatial OME-Zarr store.
	* `antartica/` 
		* `build_multisource_zarr.py` : Compiles a Zarr store (*spatial_200.zarr) from all contributing datasets for Antarctica. The chunk definition is 1024 x 1024 x 1 (x, y, t) and the resolution is 200 x 200 m.
		* `build_multisource_omezarr.py` : Compiles an OME-Zarr store (*spatial.zarr) from all contributing datasets for Antarctica. The chunk definition is 1024 x 1024 x 1 (x, y, t) and the resolution is telescopes from 200 m to 6400 m.
		* `rechunk_zarr_safe.py` : Rechunks the *spatial_200.zarr Zarr store to either a time-series optimised (chunk def: 64 x 64 x -1) Zarr or general-purpose Zarr (chunk def: 256 x 256 x 300). 
* `docs/` : Everything required to build the SHIVER readthedocs website.

## environment.yml
This code was run using Python 3.11.14. To install the necessary dependencies, you can use conda:
`conda env create -f environment.yml`

## Usage
To reproduce the dataset creation:
1. Download the raw data from the sources given in citations.py (see docs/examples/utils).
2. Update the directory paths in each script.
3. Run the scripts. Note the build*.py scripts are designed to run on a HPC, where each contributing dataset is submitted as a separate batch job.

## License
This code is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

The data are licensed under CC-BY-4.0.

## Cite
A dataset description paper for SHIVER is in preparation. Until it is published, cite it as:

> Davison, B. J. et al. (in prep). The SHeffield Ice Velocity ExploreR (SHIVER): an online tool for low latency exploration, analysis and sub-setting of unified satellite-derived ice velocity data for Earth's ice sheets. Manuscript in preparation.

The dataset itself should also be cited like this:

> Davison, B. J. (2026a). The SHeffield Ice Velocity ExploreR (SHIVER): A unified satellite-derived ice velocity dataset for Earth's ice sheets (Version v1.5.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21375859

Replace `1.5.0` with the version you actually used. The SHIVER stores are updated as new data are appended, and every store records its release in its root attributes. Each release matches a version of the [Zenodo record](https://doi.org/10.5281/zenodo.21375859), so you can always recover exactly which data were used:

```python
import xarray as xr

ds = xr.open_zarr("https://data.source.coop/uos-shiver/greenland/greenland_multisource_velocity_timeseries.zarr")
print(ds.attrs["dataset_version"], ds.attrs["last_updated"])  # e.g. 1.5.0 2026-10-16
```

The SHIVER dataset is a compilation of many published velocity datasets and so would not have been possible without those. Therefore, those contributing datasets should also be cited. Specifically, you should cite all contributing datasets that your specific query intersected. If you are extracting data programmatically, please see our notebook describing how you can generate the correct citations for your query. 

If instead you are using the SHIVER application, all exports will contain citation details for the relevant contributing datasets. This will include a table (<strong>citations_summary.csv</strong>) summarising the required citations and a text file (<strong>citations_and_usage.txt</strong>) with descriptive details. 

<em>Note: The ESA_CCI_Sentinel-1, ESA_CCI_Sentinel-2 and ESA_CCI_ERS1-2_Envisat over selected glacier sites in Greenland require separate citations for each glacier site. Our dataset compilation method combines all of those glacier sites into one datasource. Our automated citation generator will therefore provide the citation information for all glacier sites even if only one is intersected. You should remove the unnecessary citations when incorporating them into your work.</em>
