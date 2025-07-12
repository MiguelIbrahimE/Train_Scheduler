"""
ETL ‒ OSM rail infrastructure downloader
----------------------------------------

Fetches OpenStreetMap rail features for a given country / region,
persists them to GeoPackage (or any format supported by GeoPandas),
and returns the cleaned GeoDataFrame for downstream processing.

Typical use::

    python -m rail_reference_model.etl.load_osm be data/raw/rail_be.gpkg
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Download rail layers from OSM")


RAIL_TAGS = {
    "railway": [
        "rail",
        "light_rail",
        "subway",
        "tram",
        "narrow_gauge",
        "monorail",
        "funicular",
        "preserved",
        "construction",
    ]
}


def _query_place(place: str) -> gpd.GeoDataFrame:
    """
    Download all railway-related geometries for *place*.

    Parameters
    ----------
    place : str
        Any Nominatim-compatible string such as "Belgium" or "Zurich, Switzerland".

    Returns
    -------
    gpd.GeoDataFrame
    """
    logger.info("Querying OSM for %s …", place)
    gdf = ox.features_from_place(place, RAIL_TAGS)
    logger.info("Fetched %d features", len(gdf))
    return gdf.reset_index(drop=True)


@app.command()
def run(
    place: str = typer.Argument(..., help="Country / region name (Nominatim format)"),
    outfile: Path = typer.Argument(
        ...,
        help="Output file (e.g. data/raw/rail_be.gpkg or rail_jp.geojson)",
    ),
    layer: str = typer.Option(
        "rail", "--layer", help="Layer name when writing GeoPackage"
    ),
) -> None:
    """
    Download OSM rail network for *place* and save to *outfile*.
    """
    gdf = _query_place(place)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %s …", outfile)
    if outfile.suffix.lower() in {".gpkg", ".sqlite"}:
        gdf.to_file(outfile, layer=layer, driver="GPKG")
    else:
        gdf.to_file(outfile)

    logger.info("Done.")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s ‒ %(message)s"
    )
    app()
