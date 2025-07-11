"""Core dataclasses: gauge, track, train, and station profile."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List
from dataclasses import dataclass

class Gauge(Enum):
    STANDARD = 1_435
    CAPE = 1_067
    METRIC = 1_000
    BROAD = 1_676


@dataclass(frozen=True)
class TrackType:
    name: str
    gauge: Gauge
    electrified: bool
    speed_max_kmh: int
    min_radius_m: int
    capex_per_km_eur: float

    def __post_init__(self):
        if self.speed_max_kmh <= 0:
            raise ValueError("speed_max_kmh must be positive")
        if self.min_radius_m < 300:
            raise ValueError("min_radius_m unrealistically small")


@dataclass(frozen=True)
class TrainType:
    name: str
    gauge: Gauge
    capacity: int
    top_speed_kmh: int
    purchase_cost_eur: float
    opex_per_km_eur: float

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.purchase_cost_eur <= 0:
            raise ValueError("purchase_cost_eur must be > 0")


@dataclass
class StationProfile:
    name: str
    lat: float
    lon: float
    depth_m: int = 0
    tracks: int = 2
    platforms: List[int] = field(default_factory=lambda: [180])

@dataclass
class CityNode:
    """Represents a city in the network"""
    name: str
    code: str
    lat: float
    lon: float
    population: int
    elevation: float = 0
    importance_score: float = 0
    
def suggest_train_model(distance_km: float) -> TrainType:
    """
    Heuristic selection of train model based on route distance.
    You can refine this based on capacity, budget, or terrain later.
    """
    if distance_km < 50:
        return TrainType(
            name="2-car EMU",
            gauge=Gauge.STANDARD,
            capacity=250,
            top_speed_kmh=120,
            purchase_cost_eur=8_000_000,
            opex_per_km_eur=6.0
        )
    elif distance_km < 150:
        return TrainType(
            name="4-car EMU",
            gauge=Gauge.STANDARD,
            capacity=500,
            top_speed_kmh=160,
            purchase_cost_eur=14_000_000,
            opex_per_km_eur=7.5
        )
    elif distance_km < 300:
        return TrainType(
            name="6-car Intercity",
            gauge=Gauge.STANDARD,
            capacity=700,
            top_speed_kmh=200,
            purchase_cost_eur=22_000_000,
            opex_per_km_eur=9.5
        )
    else:
        return TrainType(
            name="8-car HSR",
            gauge=Gauge.STANDARD,
            capacity=900,
            top_speed_kmh=250,
            purchase_cost_eur=35_000_000,
            opex_per_km_eur=12.0
        )
