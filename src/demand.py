"""
demand.py – quick-turn demand estimator for BCPC (v0·5)

Returned unit
-------------
* A single float **ppd** = Total boardings *per direction* on an average day.
  The optimiser treats this as the north- or east-bound load; the opposite
  direction is assumed symmetrical.

Why the rewrite?
----------------
`pipeline.py` calls
    estimate_demand(population, daily_commuters, tourism_index)
and then passes the **float** to `optimise_design`.  
The earlier DemandResult object broke that contract – this version restores it
while preserving the old gravity helpers for future expansion.

Tweak-me constants live in the PUBLIC PARAMETERS block.
"""

from __future__ import annotations

import logging
from math import ceil
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("bcpc.demand")

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC PARAMETERS  (edit freely)
# ─────────────────────────────────────────────────────────────────────────────
WEEKDAY_FACTOR = 5 / 7            # share of weekdays in a mean week
WEEKEND_FACTOR = 2 / 7
WEEKEND_COMMUTER_SCALE = 0.40     # commuters who still travel on weekends
BASE_TRIPS_PER_CAP = 0.05         # discretionary trips/person/day
TOURISM_BETA = 0.40               # elasticity wrt tourism_index
INDUCED_DEMAND = 0.05             # uplift on total to reflect modal shift

# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers (kept for later OD modelling)
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_zones(population: int, granularity: int = 10_000) -> np.ndarray:
    """Split a city's population into ~equal zones.  Returns zone sizes."""
    zones = max(1, population // granularity)
    base = population // zones
    remainder = population % zones
    out = np.full(zones, base, dtype=int)
    out[:remainder] += 1
    return out


def build_gravity_matrix(zones: np.ndarray, beta: float = 0.1) -> pd.DataFrame:
    """
    Simple doubly-constrained gravity model (placeholder).

    *Not* used in the high-level `estimate_demand` yet, but kept so you can
    switch to a full 4-step model later without re-importing code.
    """
    size = len(zones)
    rng = np.random.default_rng(42)
    xy = rng.random((size, 2))                                         # fake centroids
    dists = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2) + np.eye(size)
    impedance = np.exp(-beta * dists)
    trips = np.outer(zones, zones) * impedance
    for _ in range(10):                                                # IPF
        trips *= zones / trips.sum(axis=1, keepdims=True)
        trips *= zones / trips.sum(axis=0, keepdims=True)
    return pd.DataFrame(trips)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def estimate_demand(
    population: int,
    daily_commuters: int,
    tourism_index: float | None = 0.0,
) -> float:
    """
    Return **average boardings per direction per day** for this city.

    Parameters
    ----------
    population       : residential population
    daily_commuters  : people who generate one out- and one in-bound trip on a
                       typical weekday
    tourism_index    : 0 → negligible visitors, 1 → major magnet

    The maths is deliberately back-of-the-envelope – refine as you collect real
    OD counts or mobile-phone data.
    """
    # 1 ▸ Commuters (weekday vs weekend blend)
    weekday_comm = daily_commuters
    weekend_comm = daily_commuters * WEEKEND_COMMUTER_SCALE
    commuter_ppd = (
        weekday_comm * WEEKDAY_FACTOR
        + weekend_comm * WEEKEND_FACTOR
    )

    # 2 ▸ Discretionary & tourism
    discretionary_rate = BASE_TRIPS_PER_CAP * (1 + TOURISM_BETA * (tourism_index or 0.0))
    discretionary_ppd = population * discretionary_rate

    # 3 ▸ Induced demand uplift
    gross_ppd = commuter_ppd + discretionary_ppd
    total_ppd = gross_ppd * (1 + INDUCED_DEMAND)

    logger.debug(
        "Demand → commuter=%.0f  disc=%.0f  induced=%.0f ⟹ %.0f boardings/day",
        commuter_ppd,
        discretionary_ppd,
        total_ppd - gross_ppd,
        total_ppd,
    )

    # Optimiser expects *per direction* ⇒ divide by 2 and round up
    return ceil(total_ppd / 2)
