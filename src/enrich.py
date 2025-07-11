"""
enrich.py – bring in open data (OSM boundaries, etc.)

Compatible with GeoPandas 0.14 and 1.x
"""
from __future__ import annotations

import functools
import logging
import unicodedata
from pathlib import Path
from typing import Iterable, Union

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon

from . import DATA_DIR

logger = logging.getLogger("bcpc.enrich")
ox.settings.log_console = False
ox.settings.use_cache = True

_BOUNDARY_CACHE = DATA_DIR / "_boundaries"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _cache_path(city_id: str) -> Path:
    return _BOUNDARY_CACHE / f"{city_id}.gpkg"


def _ascii(s: str) -> str:
    """Best-effort ASCII transliteration (é → e, ß → ss, …)."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _candidate_queries(city: str, iso_code: str | None) -> Iterable[str]:
    """
    Yield geocoder query strings from most to least specific.

    Example for 'Zahlé', 'LB' →
        1. 'Zahlé'
        2. 'Zahle'
        3. 'Zahlé, LB'
        4. 'Zahle, LB'
    """
    ascii_name = _ascii(city)
    yield city
    if ascii_name != city:
        yield ascii_name
    if iso_code:
        yield f"{city}, {iso_code}"
        if ascii_name != city:
            yield f"{ascii_name}, {iso_code}"


# --- inside src/enrich.py ---------------------------------------------------
def _download_boundary(city: str, iso_code: str | None) -> Polygon | MultiPolygon:
    """
    Return the best-available geometry for a city:

    1. First admin Polygon/MultiPolygon in the geocoder response.
    2. Otherwise first Point buffered to 10 km.
    3. Otherwise convex-hull the first geometry (last resort).
    """
    for q in _candidate_queries(city, iso_code):
        try:
            gdf = ox.geocoder.geocode_to_gdf(q, which_result=None)
            if gdf.empty:
                continue

            # 1 ▸ any polygon?
            for geom in gdf.geometry:
                if geom.geom_type in {"Polygon", "MultiPolygon"}:
                    logger.debug("Polygon hit for %s via '%s'", city, q)
                    return geom

            # 2 ▸ any point?
            for geom in gdf.geometry:
                if geom.geom_type == "Point":
                    logger.debug("Point hit for %s via '%s' – buffering 10 km", city, q)
                    return geom.buffer(10_000)

            # 3 ▸ fall back to convex hull of first geometry
            hull = gdf.geometry.iloc[0].convex_hull
            logger.debug("Fallback hull for %s via '%s'", city, q)
            return hull

        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"Nominatim gave no usable geometry for '{city}'")


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=64)
def get_city_boundary(city_name: str, city_id: str) -> Polygon:
    """
    Return an admin boundary polygon for the city.

    Order of battle:
      1. Return cached geometry if present.
      2. Query OSM (with accent stripping + country hint fallbacks).
      3. If only a point is returned, buffer it to a 10 km circle.
      4. Persist to cache and return.
    """
    _BOUNDARY_CACHE.mkdir(exist_ok=True, parents=True)
    cache_file = _cache_path(city_id)

    # 1 ▸ Cache
    if cache_file.exists():
        try:
            return gpd.read_file(cache_file).geometry.iloc[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Boundary cache read failed (%s) – redownloading", exc)

    # 2 ▸ Download
    iso = city_id.split("-")[0] if "-" in city_id else None
    logger.info("Downloading OSM boundary for %s", city_name)
    geom = _download_boundary(city_name, iso)

    # 3 ▸ Point → buffer
    if isinstance(geom, Point):
        logger.warning(
            "Only a point geometry found for %s – using 10 km circular buffer",
            city_name,
        )
        geom = geom.buffer(10_000)  # metres; crude but OK for demo

    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise RuntimeError(
            f"Could not obtain polygon boundary for {city_name} (got {geom.geom_type})"
        )

    # 4 ▸ Cache + return
    gpd.GeoSeries([geom]).to_file(cache_file, driver="GPKG")
    return geom if isinstance(geom, Polygon) else geom.convex_hull


# ────────────────────────────────────────────────────────────────────────────
# Candidate corridor generation (unchanged)
# ────────────────────────────────────────────────────────────────────────────
def _fallback_hex_grid(boundary: Polygon, spacing: int = 5_000) -> gpd.GeoDataFrame:
    import math
    import shapely.geometry as geom

    boundary_3857 = gpd.GeoSeries([boundary], crs=4326).to_crs(3857).iloc[0]
    minx, miny, maxx, maxy = boundary_3857.bounds
    dx = spacing * math.sqrt(3)
    dy = spacing * 1.5

    hex_lines = []
    y = miny
    row = 0
    while y < maxy + spacing:
        offset = 0 if row % 2 == 0 else dx / 2
        x = minx - offset
        while x < maxx + dx:
            center = geom.Point(x, y)
            hexagon = center.buffer(spacing, resolution=6)
            if hexagon.intersects(boundary_3857):
                hex_lines.append(hexagon.intersection(boundary_3857).boundary)
            x += dx
        y += dy
        row += 1
    return gpd.GeoDataFrame(geometry=hex_lines, crs=3857)


def build_candidate_corridors(boundary: Polygon) -> gpd.GeoDataFrame:
    try:
        import geopandas.tools as gptools

        grid = gptools.hexbin(boundary, 5_000)
        grid["geometry"] = grid.geometry.boundary
        return grid.to_crs(3857)
    except (ImportError, AttributeError, TypeError):
        logger.debug("hexbin unavailable – using fallback generator")
        return _fallback_hex_grid(boundary, 5_000)

def fetch_city_boundary(city_name: str, country: str = "Lebanon") -> gpd.GeoDataFrame:
    """
    Fetches administrative boundary of a city using OSM Nominatim.
    """
    try:
        place = f"{city_name}, {country}"
        gdf = ox.geocode_to_gdf(place)
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf.set_crs("EPSG:4326")
        return gdf
    except Exception as e:
        raise RuntimeError(f"Failed to fetch boundary for {city_name}: {e}")


__all__ = ["fetch_city_boundary"]
