"""
Lazy-loading utilities for snowsar package.

Utilities are imported on first access to avoid requiring all optional
dependencies when only a subset of functionality is needed.

Example
-------
>>> from snowsar.utils import save_pickle  # Only imports io_utils, not geopandas/rasterio/etc.
>>> save_pickle(data, "output.pkl")
"""

# Map of public names to (module, attribute) for lazy loading
_LAZY_IMPORTS = {
    # High-level context
    "InsarContext": ("insar_context", "InsarContext"),
    "build_insar_context": ("insar_context", "build_insar_context"),
    # HyP3
    "parse_unique_dates_from_hyp3_filenames": ("hyp3_utils", "parse_unique_dates_from_hyp3_filenames"),
    "parse_date_pairs_from_hyp3_filenames": ("hyp3_utils", "parse_date_pairs_from_hyp3_filenames"),
    "footprint_from_geotiffs": ("hyp3_utils", "footprint_from_geotiffs"),
    # MintPy
    "mintpy_dates_from_timeseries_h5": ("mintpy_utils", "mintpy_dates_from_timeseries_h5"),
    "mintpy_footprint_from_timeseries_h5": ("mintpy_utils", "mintpy_footprint_from_timeseries_h5"),
    # MintPy/LIDAR grids
    "get_mintpy_grid": ("lidar_utils", "get_mintpy_grid"),
    "get_geocoded_hdf5_grid": ("lidar_utils", "get_geocoded_hdf5_grid"),
    "list_hdf5_root_datasets": ("lidar_utils", "list_hdf5_root_datasets"),
    "read_hdf5_root_attributes": ("lidar_utils", "read_hdf5_root_attributes"),
    "write_mintpy_array_as_geotiff": ("lidar_utils", "write_mintpy_array_as_geotiff"),
    # Raster operations
    "resample_geotiff_to_mintpy_grid": ("lidar_utils", "resample_geotiff_to_mintpy_grid"),
    "resample_many_geotiffs": ("lidar_utils", "resample_many_geotiffs"),
    "subset_geotiff_by_bbox": ("lidar_utils", "subset_geotiff_by_bbox"),
    "read_geotiff_stack_sorted_by_date": ("lidar_utils", "read_geotiff_stack_sorted_by_date"),
    # LIDAR timeseries
    "build_lidar_timeseries_h5": ("lidar_utils", "build_lidar_timeseries_h5"),
    "extract_start_date_str": ("lidar_utils", "extract_start_date_str"),
    "compute_pearson_correlation": ("lidar_utils", "compute_pearson_correlation"),
    "cumulative_sum_through_date": ("lidar_utils", "cumulative_sum_through_date"),
    # LOS geometry
    "local_incidence_from_geometry": ("lidar_utils", "local_incidence_from_geometry"),
    "project_scalar_field_to_los": ("lidar_utils", "project_scalar_field_to_los"),
    "resample_geometry_dataset_to_raster": ("lidar_utils", "resample_geometry_dataset_to_raster"),
    "subset_radar_geometry_h5": ("lidar_utils", "subset_radar_geometry_h5"),
    # Geometry
    "get_valid_data_polygon_from_array": ("geometry", "get_valid_data_polygon_from_array"),
    # SNOTEL
    "fetch_snotel_sites": ("snotel_utils", "fetch_snotel_sites"),
    "filter_sites_by_polygon": ("snotel_utils", "filter_sites_by_polygon"),
    "fetch_snotel_timeseries": ("snotel_utils", "fetch_snotel_timeseries"),
    # IO
    "save_pickle": ("io_utils", "save_pickle"),
    "load_pickle": ("io_utils", "load_pickle"),
    # Plotting
    "plot_snotel_data": ("plotting", "plot_snotel_data"),
    "make_footprint_station_map": ("plotting", "make_footprint_station_map"),
    # NISAR utilities
    "nisar_dates_from_gunw_h5": ("nisar_utils", "nisar_dates_from_gunw_h5"),
    "nisar_footprint_from_gunw_h5": ("nisar_utils", "nisar_footprint_from_gunw_h5"),
    "nisar_union_footprints": ("nisar_utils", "nisar_union_footprints"),
    "extract_gunw_layers_to_geotiff_batch": ("nisar_utils", "extract_gunw_layers_to_geotiff_batch"),
    "download_dem_for_gunw_with_sardem": ("nisar_utils", "download_dem_for_gunw_with_sardem"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name):
    """Lazy import attributes on first access."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        # Import the module
        from importlib import import_module
        module = import_module(f".{module_name}", package="snowsar.utils")
        # Get the attribute
        attr = getattr(module, attr_name)
        # Cache it in this module for subsequent access
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'snowsar.utils' has no attribute '{name}'")


def __dir__():
    """Expose lazy exports to interactive autocomplete."""
    return sorted(set(globals()) | set(__all__))
