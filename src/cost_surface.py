"""
cost_surface.py – build a raster whose cell values represent the *relative*
construction cost of putting a new rail line through that cell.

Returns
-------
profile : dict
    rasterio-style profile (transform, CRS …)
cost : ndarray
    float32, NaN = blocked cell
"""
from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np
import rasterio
import rasterio.mask
from rasterio.enums import Resampling

log = logging.getLogger("bcpc.costsurf")


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def _downsample_slope(dem_ds, factor: int = 3) -> Tuple[dict, np.ndarray]:
    """
    Read DEM, compute slope (°), down-sample by *factor* with mean resampling.
    """
    arr = dem_ds.read(1, masked=True).filled(np.nan)
    
    # slope in degrees -------------------------------------------------------
    dy, dx = np.gradient(arr, dem_ds.res[1], dem_ds.res[0])
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    
    # target size ------------------------------------------------------------
    new_h = math.ceil(slope.shape[0] / factor)
    new_w = math.ceil(slope.shape[1] / factor)
    
    # resample
    with rasterio.Env(), rasterio.io.MemoryFile() as mem:
        vrt_opts = {
            "driver": "GTiff",  # Changed from VRT to GTiff
            "width": new_w,
            "height": new_h,
            "count": 1,  # Explicitly set band count
            "dtype": "float32",
            "transform": dem_ds.transform * dem_ds.transform.scale(factor, factor),
            "crs": dem_ds.crs,
            "nodata": np.nan,
        }
        
        with mem.open(**vrt_opts) as dst:
            # Ensure slope is float32 and write it
            slope_data = slope.astype("float32")
            dst.write(slope_data, indexes=1)
        
        with mem.open() as ds:
            slope_ds = ds.read(
                1,
                out_shape=(new_h, new_w),
                resampling=Resampling.average,
                masked=True,
            )
            
            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "height": new_h,
                "width": new_w,
                "crs": dem_ds.crs,
                "transform": dem_ds.transform * dem_ds.transform.scale(factor, factor),
                "nodata": np.nan,
            }
    
    return profile, slope_ds.filled(np.nan)


# ────────────────────────────────────────────────────────────────────────────
# public builder
# ────────────────────────────────────────────────────────────────────────────
def build_cost_surface(boundary_gdf, dem_ds):
    """
    Return (profile, cost_array) masked to *boundary_gdf*.

    Cost model (v0·1)
    -----------------
    cost = 1 + (slope° / 3)²

    • flat ground → ≈1
    • 6 % gradient → ≈5×
    • 10 % → ≈12×
    """
    profile, slope = _downsample_slope(dem_ds, factor=3)
    
    # crop mask --------------------------------------------------------------
    mask_geom = boundary_gdf.unary_union
    
    # Create transform for the downsampled raster
    transform = profile["transform"]
    
    # Create a temporary raster with the slope data for masking
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(slope, 1)
        
        with memfile.open() as dataset:
            mask_arr, mask_transform = rasterio.mask.mask(
                dataset, [mask_geom], crop=False, nodata=np.nan, filled=True
            )
    
    cost = 1.0 + (slope / 3.0) ** 2
    cost[mask_arr[0] == profile["nodata"]] = np.nan  # outside study area
    
    log.debug("Cost surface built – %.1f %% valid cells", 100 * np.isfinite(cost).mean())
    
    return profile, cost


__all__ = ["build_cost_surface"]