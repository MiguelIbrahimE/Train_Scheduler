# terrain.py – DEM fetch & utilities (OpenTopography edition, v0·7)
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import rasterio.enums
import rasterio.windows
import requests
from rasterio.io import DatasetReader, MemoryFile
from shapely.geometry import box

from src import DATA_DIR                     # <-- unchanged

logger = logging.getLogger("bcpc.terrain")

_DEM_CACHE = DATA_DIR / "_dem"
_DEM_CACHE.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
OT_API   = "https://portal.opentopography.org/API/globaldem"
OT_KEY   = os.getenv("OT_API_KEY", "153e670200e6b3568ff813c994fda446")
OT_TIMEO = 60  # seconds

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _bbox_from_gdf(gdf) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    pad = 0.05  # deg padding so edges aren’t cut off
    return maxy + pad, miny - pad, minx - pad, maxx + pad  # N, S, W, E


def _hash_bbox(n: float, s: float, w: float, e: float, demtype: str) -> str:
    m = hashlib.md5()
    m.update(f"{demtype}:{n:.4f},{s:.4f},{w:.4f},{e:.4f}".encode())
    return m.hexdigest()[:16]


def _download_dem(n: float, s: float, w: float, e: float, demtype: str) -> Path:
    cache_name = f"{_hash_bbox(n, s, w, e, demtype)}_{demtype}.tif"
    cache_path = _DEM_CACHE / cache_name
    if cache_path.exists():
        logger.debug("DEM cache hit (%s)", cache_path.name)
        return cache_path

    params = dict(
        demtype=demtype, south=s, north=n, west=w, east=e,
        outputFormat="GTiff", API_Key=OT_KEY
    )
    logger.info("Fetching %s DEM  (%4.2f,%4.2f,%4.2f,%4.2f)", demtype, s, n, w, e)
    with requests.get(OT_API, params=params, stream=True, timeout=OT_TIMEO) as r:
        r.raise_for_status()
        if r.headers.get("Content-Type", "").startswith("text"):
            raise RuntimeError(r.text.strip())

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        for chunk in r.iter_content(chunk_size=1 << 16):
            tmp.write(chunk)
        tmp.close()
        Path(tmp.name).rename(cache_path)
    return cache_path


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def _flat_memory_dem() -> DatasetReader:
    """Return an in-memory 100 × 100 raster filled with zeros (WGS-84)."""
    transform = rasterio.transform.from_origin(-180, 90, 0.01, 0.01)
    data = np.zeros((100, 100), dtype=np.float32)

    mem = MemoryFile()
    with mem.open(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as ds:
        ds.write(data, 1)
    return mem.open()       # reopen in read-only mode


def load_dem(boundary_gdf) -> DatasetReader:
    n, s, w, e = _bbox_from_gdf(boundary_gdf)
    for dem in ("SRTMGL1_E", "ASTER30m"):
        try:
            disk_path = _download_dem(n, s, w, e, dem)
            with rasterio.open(disk_path) as src:
                profile = src.profile
                data = src.read()

            memfile = MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(data)

            return memfile.open()
        except Exception as exc:
            logger.warning("%s fetch failed – %s", dem, exc)

    logger.error("DEM fetch failed – using synthetic flat DEM (0 m)")
    return _flat_memory_dem()




def slope_percent(ds: DatasetReader) -> np.ndarray:
    """
    %-slope raster (same shape as DEM).  For tiny rasters (<2 px per axis)
    return an array of zeros to avoid np.gradient errors.
    """
    band = ds.read(1, masked=True).filled(np.nan)
    if min(band.shape) < 2:
        return np.zeros_like(band, dtype=np.float32)

    dy, dx = np.gradient(band, ds.res[1], ds.res[0])
    slope_rad = np.arctan(np.hypot(dx, dy))
    return np.degrees(slope_rad) * 100 / 45.0  # ≃ % slope

# ---------------------------------------------------------------------------
def ruggedness_index(ds: DatasetReader) -> float:
    slope_pct = slope_percent(ds)
    mean_deg  = np.nanmean(slope_pct) * 0.45        # % → °
    return max(0.0, min(mean_deg / 45.0, 1.0))
