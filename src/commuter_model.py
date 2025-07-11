"""
commuter_model.py – back-of-the-envelope commuter estimator.

Method
------
* Gravity model: commuters ∝ pop_i * pop_j / distance²
* We treat the *target city* as a node with mass = its population and
  interact it with **all the other rows** loaded in memory.
"""

from __future__ import annotations
from math import ceil
from typing import Sequence

import numpy as np
from geopy.distance import great_circle

from .scenario_io import ScenarioRow
from .enrich import get_city_boundary


def _city_centroid(row: ScenarioRow):
    geom = get_city_boundary(row.city_name, row.city_id)
    return geom.centroid.y, geom.centroid.x  # lat, lon


def estimate_commuters(row: ScenarioRow, all_rows: Sequence[ScenarioRow]) -> int:
    pop_i = row.population
    lat_i, lon_i = _city_centroid(row)
    total = 0.0

    for other in all_rows:
        if other.city_id == row.city_id:
            continue
        pop_j = other.population
        lat_j, lon_j = _city_centroid(other)
        d_km = great_circle((lat_i, lon_i), (lat_j, lon_j)).km or 1.0
        total += pop_i * pop_j / d_km**2

    # scale constant so Beirut≈300k commuters (tune as you wish)
    SCALE = 3e-10
    return ceil(total * SCALE)
