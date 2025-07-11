# ── src/city_pipeline.py ────────────────────────────────────────────────────
"""
city_pipeline.py – run the BCPC stack for **one** city (v0·9)

* Imputes `daily_commuters`    via estimate_commuters(...)
* Imputes `terrain_ruggedness` via mean-slope raster (ruggedness_index)
* Accepts the chosen `track` and `train` as parameters
* Returns (NetworkDesign, CostBreakdown, Path to artefact)

Called by the new global planner for every corridor-section,
but still runnable stand-alone for unit tests.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely.geometry as geom
from shapely.geometry import MultiPolygon, Polygon

from . import OUTPUT_DIR, logger                       # root logger config
from .cost import CostBreakdown, estimate_cost
from .demand import estimate_demand
from .enrich import get_city_boundary
from .scenario_io import ScenarioRow, save_geojson
from .models import TrackType, TrainType
from .optimise import NetworkDesign, optimise_design
from .routing import RoutingError, trace_route
from .terrain import load_dem, ruggedness_index

# module-level logger
log = logging.getLogger("bcpc.city_pipeline")
log.setLevel(logging.INFO)


# ─────────────────────────── helper utils ──────────────────────────────────
def _nx_to_gdf(g: nx.Graph) -> gpd.GeoDataFrame:
    """Convert a Shapely-edge graph to GeoPandas."""
    if g.number_of_edges() == 0:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    crs = g.graph.get("crs", "EPSG:4326")
    rows = [
        dict(u=u, v=v,
             length_km=data["geometry"].length / 1_000,
             geometry=data["geometry"])
        for u, v, data in g.edges(data=True)
    ]
    return gpd.GeoDataFrame(rows, crs=crs)


def _select_extreme_points(
    geom_in: Polygon | MultiPolygon
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Two boundary coords with max Euclidean separation (fast heuristic)."""
    polys = [geom_in] if isinstance(geom_in, Polygon) else list(geom_in.geoms)
    coords: list[Tuple[float, float]] = []
    for p in polys:
        coords.extend(p.exterior.coords)

    # down-sample long rings (O(N²) search otherwise)
    coords = coords[:: max(1, len(coords) // 500)]

    xy = np.asarray(coords)
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return tuple(coords[i]), tuple(coords[j])


# ─────────────────────────────── main API ──────────────────────────────────
def run_city_pipeline(
    row: ScenarioRow,
    all_rows: Sequence[ScenarioRow],
    track: TrackType,
    train: TrainType,
) -> tuple[NetworkDesign, CostBreakdown, Path]:
    """
    Core per-city routine.

    Parameters
    ----------
    row       : one ScenarioRow (city)
    all_rows  : full list (needed for commuter imputation)
    track     : TrackType already selected by global planner
    train     : TrainType already selected by global planner
    """
    log.info("↳ %s – start", row.city_name)

    # 1 ▸ boundary & DEM ---------------------------------------------------
    boundary = get_city_boundary(row.city_name, row.city_id)
    gdf_boundary = gpd.GeoSeries([boundary], crs=4326)

    dem = load_dem(gdf_boundary)

    # ── ruggedness auto-fill
    if row.terrain_ruggedness is None:
        row.terrain_ruggedness = ruggedness_index(dem)
        log.debug("Imputed ruggedness %.2f for %s",
                  row.terrain_ruggedness, row.city_name)

    # 2 ▸ provisional termini ---------------------------------------------
    origin, dest = _select_extreme_points(boundary)

    # 3 ▸ routing ----------------------------------------------------------
    try:
        line = trace_route(origin, dest, gdf_boundary, dem, track)
    except RoutingError as exc:
        log.warning("Routing failed: %s – fallback straight line", exc)
        line = geom.LineString([origin, dest])

    G = nx.Graph(crs="EPSG:4326")
    G.add_edge(line.coords[0], line.coords[-1],
               geometry=line, weight=line.length / 1_000)

    # 4 ▸ commuter auto-fill ----------------------------------------------
    if row.daily_commuters is None:
        row.daily_commuters = estimate_commuters(row, all_rows)
        log.debug("Imputed %d commuters for %s",
                  row.daily_commuters, row.city_name)

    demand_ppd = estimate_demand(
        row.population, row.daily_commuters, row.tourism_index
    )

    design = optimise_design(demand_ppd, row.budget_total_eur)
    # overwrite graph/rolling-stock with caller-chosen versions
    design.graph, design.track, design.train = G, track, train

    # 5 ▸ cost breakdown ---------------------------------------------------
    km_track = sum(d["geometry"].length
                   for _, _, d in G.edges(data=True)) / 1_000
    cost_br = estimate_cost(km_track, len(G.nodes), 1, 5, underground_ratio=0)
    design.cost_eur = cost_br.total()

    # 6 ▸ export artefact --------------------------------------------------
    if G.number_of_edges() == 0:
        artefact = OUTPUT_DIR / f"{row.city_id}_EMPTY.txt"
        artefact.write_text("Network empty – budget insufficient.\n")
    else:
        artefact = OUTPUT_DIR / f"{row.city_id}_tracks.geojson"
        save_geojson(_nx_to_gdf(G), artefact)

    # 7 ▸ KPI log ----------------------------------------------------------
    log.info("%s ✓ CAPEX €%.1f M | %d stations | %s pax/day",
             row.city_name,
             design.cost_eur / 1e6,
             len(design.graph.nodes),
             f"{design.ridership_daily:,.0f}")

    return design, cost_br, artefact
