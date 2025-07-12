"""
Edge-level feature extractor.

Reads the GraphML built by *build_graph.py* and writes a tidy
Parquet table with numeric features suitable for ML.

Features
--------
u, v              : node IDs
length_m          : geodesic length
straight_m        : straight-line distance between endpoints
curvature         : length_m / straight_m  (≥1)
maxspeed_kmh      : OSM tag, converted to float or NaN

CLI
---
python -m rail_reference_model.features.vectorize \
       data/processed/rail_be.graphml \
       data/processed/rail_be_edges.parquet
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import networkx as nx
import pandas as pd
import pyproj
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Vectorise graph edges into ML features")

_GEOD = pyproj.Geod(ellps="WGS84")
_KMH_RE = re.compile(r"(\d+)")


def _straight_distance(u_attrs: Dict[str, Any], v_attrs: Dict[str, Any]) -> float:
    """Great-circle distance between two graph nodes in metres."""
    lon1, lat1 = u_attrs["x"], u_attrs["y"]
    lon2, lat2 = v_attrs["x"], v_attrs["y"]
    _, _, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
    return dist


def _parse_maxspeed(val: Any) -> float | None:
    """
    Convert OSM maxspeed string / int → km/h float.
    Examples: '140', '140.0', '100 mph', '100;120'
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    match = _KMH_RE.search(val)
    return float(match.group(1)) if match else None


def _edge_features(G: nx.MultiDiGraph) -> pd.DataFrame:
    rows = []
    for u, v, data in G.edges(data=True):
        u_attr, v_attr = G.nodes[u], G.nodes[v]
        straight_m = _straight_distance(u_attr, v_attr)
        length_m = float(data.get("length_m", straight_m))
        rows.append(
            dict(
                u=u,
                v=v,
                length_m=length_m,
                straight_m=straight_m,
                curvature=length_m / straight_m if straight_m > 0 else 1.0,
                maxspeed_kmh=_parse_maxspeed(data.get("maxspeed")),
            )
        )
    return pd.DataFrame(rows)


@app.command()
def run(
    graph_file: Path = typer.Argument(..., help="Input GraphML built by build_graph.py"),
    outfile: Path = typer.Argument(..., help="Output Parquet/CSV with edge features"),
) -> None:
    """
    Extract numeric edge features from *graph_file* to *outfile*.
    """
    G = nx.read_graphml(graph_file)
    df = _edge_features(G)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(outfile, index=False)
    else:
        df.to_csv(outfile, index=False)

    logger.info("Wrote %s (%d edge rows)", outfile, len(df))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s ‒ %(message)s"
    )
    app()
