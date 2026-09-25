# Cite

The recommended approach to citing these data in your work is:

> "Ice velocity estimates were extracted from the Sheffield Ice Velocity ExploreR (SHIVER) unified dataset (Davison, 2026; Davison et al., 2026) and included the contributing datasets detailed in Table X."

Where Davison et al. 2026 is a dataset description paper for SHIVER (in preparation). Until it is published, cite it as:

> Davison, B. J. et al. (in prep). The SHeffield Ice Velocity ExploreR (SHIVER): an online tool for low latency exploration, analysis and sub-setting of unified satellite-derived ice velocity data for Earth's ice sheets. Manuscript in preparation.

And Davison (2026) is the citation for the dataset itself:

> Davison, B. J. (2026a). The SHeffield Ice Velocity ExploreR (SHIVER): A unified satellite-derived ice velocity dataset for Earth's ice sheets (Version v1.4.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21375859

Replace `1.4.1` with the version you actually used. The SHIVER stores are updated as new data are appended, and every store records its release in its root attributes. Each release matches a version of the [Zenodo record](https://doi.org/10.5281/zenodo.21375859), so you can always recover exactly which data were used:

```python
import xarray as xr

ds = xr.open_zarr("https://data.source.coop/uos-shiver/greenland/greenland_multisource_velocity_timeseries.zarr")
print(ds.attrs["dataset_version"], ds.attrs["last_updated"])  # e.g. 1.4.1 2026-08-20
```

The SHIVER dataset is a compilation of many published velocity datasets and so would not have been possible without those. Therefore, those contributing datasets should also be cited. Specifically, you should cite all contributing datasets that your specific query intersected. If you are extracting data programmatically, please see our notebook describing how you can generate the correct citations for your query. 

If instead you are using the SHIVER application, all exports will contain citation details for the relevant contributing datasets. This will include a table (<strong>citations_summary.csv</strong>) summarising the required citations and a text file (<strong>citations_and_usage.txt</strong>) with descriptive details. 

<em>Note: The ESA_CCI_Sentinel-1, ESA_CCI_Sentinel-2 and ESA_CCI_ERS1-2_Envisat over selected glacier sites in Greenland require separate citations for each glacier site. Our dataset compilation method combines all of those glacier sites into one datasource. Our automated citation generator will therefore provide the citation information for all glacier sites even if only one is intersected. You should remove the unnecessary citations when incorporating them into your work.</em>


