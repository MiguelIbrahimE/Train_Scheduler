"""
routing.py – basic corridor generator                           v0·2
────────────────────────────────────────────────────────────────────
If no railway exists, we:
  ① fetch the local road network inside the study-area polygon,
  ② snap the two termini to that graph, and
  ③ run A* with edge-length weights to get a plausible alignment.

Later on you can swap the `*_route_osm()` helper with a raster
least-cost-path that also uses the cost-surface – the public API below
will not change.

Public symbols
--------------
RoutingError           – exception the pipeline expects
trace_route(...)       – returns shapely LineString (lon/lat WGS-84)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Geod
from shapely.geometry import LineString, Point

log = logging.getLogger("bcpc.routing")
_GEOD = Geod(ellps="WGS84")                    # thread-safe geodesic helper


# ────────────────────────────────────────────────────────────────────────────
# Exceptions
# ────────────────────────────────────────────────────────────────────────────
class RoutingError(RuntimeError):
    """Raised when no feasible alignment can be found."""


# ────────────────────────────────────────────────────────────────────────────
# internal helpers
# ────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=8)
def _road_graph(boundary_poly) -> nx.MultiDiGraph:
    """
    Cache OSM road graphs – avoids hitting the Overpass API for every route.
    """
    log.info("Fetching OSM drive graph …")
    G = ox.graph_from_polygon(boundary_poly, network_type="drive", retain_all=True)
    if G.number_of_edges() == 0:
        raise RoutingError("OSM returned an empty road graph")
    return G


def _route_osm(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    boundary: gpd.GeoSeries,
) -> LineString:
    """
    Shortest-path on the OSM drive graph (length-weighted).
    """
    boundary_poly = boundary.unary_union
    G = _road_graph(boundary_poly)

    # snap termini -----------------------------------------------------------
    try:
        orig_id = ox.distance.nearest_nodes(G, origin[0], origin[1])
        dest_id = ox.distance.nearest_nodes(G, dest[0], dest[1])
    except Exception as exc:  # noqa: BLE001
        raise RoutingError(f"Failed to snap termini: {exc}") from exc

    # A* / Dijkstra ----------------------------------------------------------
    try:
        route_nodes = nx.shortest_path(G, orig_id, dest_id, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise RoutingError("No path in road graph") from exc

    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in route_nodes]
    line = LineString(coords)

    # sanity check – if absurdly indirect fallback to straight chord ----------
    gc_dist = abs(_GEOD.line_length(
        [origin[0], dest[0]],
        [origin[1], dest[1]],
    ))
    if gc_dist == 0 or line.length > 5 * gc_dist:
        log.warning("Road route ≫ great-circle – using straight chord")
        line = LineString([origin, dest])

    return line


# ────────────────────────────────────────────────────────────────────────────
# public API
# ────────────────────────────────────────────────────────────────────────────
def trace_route(
    origin_lonlat: Tuple[float, float],
    dest_lonlat: Tuple[float, float],
    boundary_gdf: gpd.GeoSeries,
    *_,
    **__,
) -> LineString:
    """
    Trace a route across terrain using a raster-based least-cost path.

    Uses DEM + slope penalty instead of road graph.
    """
    from src.terrain import load_dem
    from src.cost_surface import build_cost_surface
    import rasterio
    from rasterio.transform import rowcol
    from scipy.sparse.csgraph import dijkstra
    from scipy.sparse import csr_matrix
    import numpy as np
    from shapely.geometry import LineString

    # 1. Load DEM for bounding box
    dem = load_dem(boundary_gdf)

    # 2. Build terrain-based cost surface
    profile, cost = build_cost_surface(boundary_gdf, dem)

    if not np.isfinite(cost).any():
        raise RoutingError("Cost surface is entirely invalid")

    # 3. Convert lat/lon to raster indices
    transform = profile["transform"]
    try:
        start_rc = rowcol(transform, origin_lonlat[0], origin_lonlat[1])
        end_rc   = rowcol(transform, dest_lonlat[0], dest_lonlat[1])
    except Exception:
        raise RoutingError("Failed to convert coordinates to raster indices")

    # Bounds check
    rows, cols = cost.shape
    if not (0 <= start_rc[0] < rows and 0 <= start_rc[1] < cols):
        raise RoutingError(f"Start point {origin_lonlat} is outside raster bounds")
    if not (0 <= end_rc[0] < rows and 0 <= end_rc[1] < cols):
        raise RoutingError(f"End point {dest_lonlat} is outside raster bounds")

    # 4. Prepare graph for Dijkstra over raster
    indices = np.arange(rows * cols).reshape(rows, cols)

    valid_mask = np.isfinite(cost)
    
    # Build edge lists
    row_indices = []
    col_indices = []
    weights = []

    for r in range(rows):
        for c in range(cols):
            if not valid_mask[r, c]:
                continue
            current_idx = indices[r, c]
            
            # 4-connectivity
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and valid_mask[rr, cc]:
                    neighbor_idx = indices[rr, cc]
                    row_indices.append(current_idx)
                    col_indices.append(neighbor_idx)
                    weights.append(cost[rr, cc])

    if not row_indices:
        raise RoutingError("No valid pathable pixels in raster")

    # Create sparse matrix
    graph = csr_matrix(
        (weights, (row_indices, col_indices)), 
        shape=(rows * cols, rows * cols)
    )

    # 5. Solve shortest path
    start_idx = indices[start_rc]
    end_idx = indices[end_rc]

    dist_matrix, predecessors = dijkstra(
        csgraph=graph, 
        directed=False, 
        indices=start_idx, 
        return_predecessors=True
    )

    if np.isinf(dist_matrix[end_idx]):
        raise RoutingError("No path found in raster cost surface")

    # 6. Reconstruct path
    path = []
    i = end_idx
    while i != start_idx:
        r, c = divmod(i, cols)
        x, y = transform * (c + 0.5, r + 0.5)
        path.append((x, y))
        i = predecessors[i]
        if i == -9999:
            raise RoutingError("Path reconstruction failed")

    # Add start point
    r, c = divmod(start_idx, cols)
    x, y = transform * (c + 0.5, r + 0.5)
    path.append((x, y))

    path.reverse()
    return LineString(path)


__all__ = ["RoutingError", "trace_route"]