"""
Terrain-helper utilities used by the visualisation layer.

The module exposes:

• InfrastructureType – how the track is engineered on each sub-segment  
• TerrainComplexity – simple taxonomy of the surrounding topography  
• classify_segment(line, dem_path) – splits a LineString into fine-grained
  sub-segments and assigns an InfrastructureType to each one, using a DEM.

The current implementation is intentionally lightweight – it keeps every
consecutive pair of vertices as one segment and always tags it as GROUND.
Extend `classify_segment` with a proper DEM lookup and slope/height logic
when you have the elevation raster in `dem_path`.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString


class InfrastructureType(Enum):
    GROUND = "ground"
    TUNNEL = "tunnel"
    VIADUCT = "viaduct"


class TerrainComplexity(Enum):
    FLAT = "flat"
    ROLLING = "rolling"
    MOUNTAINOUS = "mountainous"


def classify_segment(
    line: LineString,
    dem_path: str,
    *,
    slope_tunnel_pct: float = 3.0,
    slope_viaduct_pct: float = 1.5,
) -> List[Tuple[LineString, InfrastructureType]]:
    """
    Very-thin stub – keeps every edge as a sub-segment and tags it GROUND.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Already densified route geometry.
    dem_path : str
        Path to a DEM (GeoTIFF etc.).  **Unused for now – plug your own logic.**
    slope_tunnel_pct / slope_viaduct_pct : float
        Design thresholds (grade %) that would trigger tunnel / viaduct.

    Returns
    -------
    list[(LineString, InfrastructureType)]
    """
    coords = list(line.coords)
    segments: List[Tuple[LineString, InfrastructureType]] = []

    for idx in range(len(coords) - 1):
        segment = LineString([coords[idx], coords[idx + 1]])

        # ── Placeholder: every sub-segment is on ground ───────────────────────
        tag = InfrastructureType.GROUND
        # TODO: sample DEM, compute slope and switch `tag` accordingly.

        segments.append((segment, tag))

    return segments
