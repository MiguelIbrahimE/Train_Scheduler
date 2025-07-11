"""
cost.py – Parametric cost estimation for tracks, stations, rolling stock.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from shapely.geometry import LineString
from src.models import TrainType

logger = logging.getLogger("bcpc.cost")


@dataclass
class CostBreakdown:
    track_eur: float
    station_eur: float
    yard_eur: float
    rolling_stock_eur: float

    def total(self) -> float:
        return sum(vars(self).values())

    def __str__(self):
        return (
            f"Track: €{self.track_eur:,.0f}, "
            f"Stations: €{self.station_eur:,.0f}, "
            f"Yards: €{self.yard_eur:,.0f}, "
            f"Trains: €{self.rolling_stock_eur:,.0f}, "
            f"Total: €{self.total():,.0f}"
        )


def estimate_cost(
    km_track: float,
    n_stations: int,
    n_yards: int,
    trainsets: int,
    underground_ratio: float = 0.15,
) -> CostBreakdown:
    """
    Rough CAPEX estimation. Underground multiplier and unit costs can be adjusted.
    """
    COST_PER_KM_AT_GRADE = 15_000_000
    UNDERGROUND_MULTIPLIER = 5.0
    COST_PER_STATION = 20_000_000
    COST_PER_YARD = 100_000_000
    COST_PER_TRAINSET = 12_000_000

    track_cost = km_track * COST_PER_KM_AT_GRADE * (
        (1 - underground_ratio) + underground_ratio * UNDERGROUND_MULTIPLIER
    )
    station_cost = n_stations * COST_PER_STATION
    yard_cost = n_yards * COST_PER_YARD
    fleet_cost = trainsets * COST_PER_TRAINSET

    breakdown = CostBreakdown(
        track_eur=track_cost,
        station_eur=station_cost,
        yard_eur=yard_cost,
        rolling_stock_eur=fleet_cost,
    )
    logger.debug("Cost breakdown: %s", breakdown)
    return breakdown


def estimate_route_cost(
    route: LineString,
    train: TrainType,
    stations: int = 2,
    yards: int = 0,
    underground_ratio: float = 0.15,
) -> CostBreakdown:
    """
    Estimate full cost of a route given a LineString and train model.
    """
    if not isinstance(route, LineString):
        raise ValueError("Route must be a shapely LineString")

    km_length = route.length * 111  # approx degrees to km (adjust if needed)
    trainsets = max(1, round(km_length / 50))  # one trainset per 50 km by default

    logger.info(
        f"Estimating cost for route of {km_length:.1f} km with {trainsets} trainsets"
    )

    return estimate_cost(
        km_track=km_length,
        n_stations=stations,
        n_yards=yards,
        trainsets=trainsets,
        underground_ratio=underground_ratio,
    )
