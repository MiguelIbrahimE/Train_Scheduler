"""
Convert OSM rail geometries to a routable NetworkX graph.

Input:  GeoPackage / GeoJSON created by `etl.load_osm.run`
Output: GraphML (or any NetworkX-supported format) ready for GNN embedding.

CLI
---
python -m rail_reference_model.features.build_graph \
       data/raw/rail_be.gpkg \
       data/processed/rail_be.graphml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import networkx as nx
import pyproj
import typer
from shapely.geometry import LineString, MultiLineString

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Build NetworkX rail graphs")

_GEOD = pyproj.Geod(ellps="WGS84")


def _segmentize(
    geom: LineString | MultiLineString,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Yield (start-coord, end-coord, great-circle length [m]) for every
    consecutive point pair inside *geom*.
    """
    if isinstance(geom, MultiLineString):
        for part in geom.geoms:
            yield from _segmentize(part)
    else:
        coords = list(geom.coords)
        for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
            _, _, dist_m = _GEOD.inv(lon1, lat1, lon2, lat2)
            yield (lon1, lat1), (lon2, lat2), dist_m


def _make_graph(gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """
    Convert rail LineStrings in *gdf* → MultiDiGraph (undirected).
    Nodes are unique (lon, lat) tuples rounded to 7 decimals.
    Edge attributes: length_m, maxspeed (if tag present).
    """
    node_id: Dict[Tuple[float, float], int] = {}
    G = nx.MultiDiGraph()

    for idx, row in gdf.iterrows():
        geom = row.geometry
        maxspeed = row.get("maxspeed")

        for coord_u, coord_v, dist_m in _segmentize(geom):
            for coord in (coord_u, coord_v):
                if coord not in node_id:
                    node_id[coord] = len(node_id)
                    G.add_node(
                        node_id[coord],
                        x=coord[0],
                        y=coord[1],
                    )

            u, v = node_id[coord_u], node_id[coord_v]
            G.add_edge(
                u,
                v,
                length_m=dist_m,
                maxspeed=maxspeed,
                osmid=idx,
            )

    logger.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


@app.command()
def run(
    infile: Path = typer.Argument(..., help="GeoPackage / GeoJSON with rail geometries"),
    outfile: Path = typer.Argument(..., help="Destination GraphML file"),
) -> None:
    """
    Build a NetworkX graph from *infile* and save to *outfile*.
    """
    # build_graph.py  (inside run())
    gdf = gpd.read_file(infile)

    # keep only lines
    gdf = gdf[gdf.geometry.type.isin({"LineString", "MultiLineString"})]

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)


    G = _make_graph(gdf)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    # drop attributes whose value is None (GraphML forbids nulls)
    for _, _, data in G.edges(data=True):
        null_keys = [k for k, v in data.items() if v is None]
        for k in null_keys:
            del data[k]

    nx.write_graphml(G, outfile)
    logger.info("Graph saved to %s", outfile)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s ‒ %(message)s"
    )
    app()
