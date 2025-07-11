"""
optimise.py – capacity-aware mixed-integer optimiser (v0·7)

Key upgrades
------------
✓ Uses OR-Tools/SCIP to pick ONE track, ONE train type and an integer fleet size
  that (i) meets the daily ridership target and (ii) stays within the budget.
✓ Gauge compatibility enforced (metric trains cannot run on standard track).
✓ Falls back transparently to the old greedy loop if OR-Tools is missing.
✓ Logs a concise CAPEX breakdown so the pipeline can print clean KPIs.

Assumptions (override at call-site if you like)
------------------------------------------------
* Service span: 18 hours/day.
* Target headway on the core link: 30 minutes (2 trains/hr *directional*).
* Average in-service load factor: 70 %.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
from typing import List

import networkx as nx

try:
    # OR-Tools ships with PyPI wheel; SCIP is the default LP/MIP backend.
    from ortools.linear_solver import pywraplp as _or_tools

    _ORTOOLS_OK = True
except ModuleNotFoundError:  # pragma: no cover – CI has OR-Tools
    _ORTOOLS_OK = False

from .models import Gauge, TrackType, TrainType

logger = logging.getLogger("bcpc.optimise")


# ---------------------------------------------------------------------------#
# Catalogue (would normally live in YAML)                                    #
# ---------------------------------------------------------------------------#
STD_TRACK = TrackType(
    name="Std-Catenary-160",
    gauge=Gauge.STANDARD,
    electrified=True,
    speed_max_kmh=160,
    min_radius_m=1_200,
    capex_per_km_eur=12_000_000,
)
METRIC_TRACK = TrackType(
    name="Metric-Diesel-100",
    gauge=Gauge.METRIC,
    electrified=False,
    speed_max_kmh=100,
    min_radius_m=800,
    capex_per_km_eur=7_000_000,
)

EMU = TrainType(
    name="4-car EMU",
    gauge=Gauge.STANDARD,
    capacity=400,
    top_speed_kmh=160,
    purchase_cost_eur=10_000_000,
    opex_per_km_eur=8,
)
DMU = TrainType(
    name="2-car DMU",
    gauge=Gauge.METRIC,
    capacity=140,
    top_speed_kmh=100,
    purchase_cost_eur=4_000_000,
    opex_per_km_eur=6,
)

TRACK_CHOICES: List[TrackType] = [STD_TRACK, METRIC_TRACK]
TRAIN_CHOICES: List[TrainType] = [EMU, DMU]

# ---------------------------------------------------------------------------#
# Dataclass returned to the pipeline                                         #
# ---------------------------------------------------------------------------#


@dataclass
class NetworkDesign:
    graph: nx.Graph
    track: TrackType
    train: TrainType
    ridership_daily: float
    cost_eur: float


# ---------------------------------------------------------------------------#
# Helper – stub “alignment” (1 km star) used by optimiser                    #
# ---------------------------------------------------------------------------#


def _demo_star_graph() -> nx.Graph:
    """Return a 1 km hub-and-spoke star centred on (0,0) – placeholder."""
    g = nx.Graph()
    # Two 0.5 km spokes – cheap stand-in until routing provides real geometry
    g.add_edge((0, 0), (0.5, 0), weight=0.5)
    g.add_edge((0, 0), (-0.5, 0), weight=0.5)
    return g


# ---------------------------------------------------------------------------#
# Fleet sizing utility                                                       #
# ---------------------------------------------------------------------------#


def _fleet_size(
    daily_pax: float,
    train: TrainType,
    headway_min: int = 30,
    service_hours: int = 18,
    load_factor: float = 0.70,
) -> int:
    """
    Compute the smallest integer fleet capable of carrying `daily_pax`.

    Parameters
    ----------
    daily_pax : float
        Projected boardings per *direction* per day.
    train     : TrainType
    headway_min : int, default 30
        One train every `headway_min` minutes in each direction.
    service_hours : int, default 18
        How long the service runs daily.
    load_factor : float, default 0.70
        Average occupied seats ÷ total seats for planning purposes.
    """
    trains_per_hour = 60 / headway_min
    total_cycles = trains_per_hour * service_hours
    seats_per_cycle = train.capacity * load_factor
    # pax / (seats per train * cycles) → trains needed in circulation
    needed = daily_pax / (seats_per_cycle * total_cycles)
    # Add one spare set for maintenance rotation (rule-of-thumb 10 %)
    return max(1, ceil(1.1 * needed))


# ---------------------------------------------------------------------------#
# Main optimiser                                                             #
# ---------------------------------------------------------------------------#


def optimise_design(demand_ppd: float, budget_eur: float) -> NetworkDesign:
    """
    Choose a (track, train, fleet_size) combo that meets `demand_ppd`
    *boardings per day per direction* and stays within the capital budget.

    Returns a NetworkDesign with a 1 km placeholder graph; the caller typically
    overwrites `.graph` once the real alignment is traced.
    """
    graph = _demo_star_graph()
    km_route = sum(d["weight"] for _, _, d in graph.edges(data=True))

    if _ORTOOLS_OK:  # ------------------------------------------------ MIP --
        solver = _or_tools.Solver.CreateSolver("SCIP")
        if solver is None:  # pragma: no cover
            logger.warning("SCIP backend missing – falling back to greedy")
            return _greedy(demand_ppd, budget_eur, graph, km_route)

        # Binary choice vars
        x_track = {i: solver.BoolVar(f"x_track_{i}") for i in range(len(TRACK_CHOICES))}
        y_train = {k: solver.BoolVar(f"y_train_{k}") for k in range(len(TRAIN_CHOICES))}
        # Integer fleet size (upper-bounded so search space stays tight)
        n_sets = solver.IntVar(0, 50, "fleet_size")

        # Exactly ONE track & ONE train
        solver.Add(solver.Sum(x_track.values()) == 1)
        solver.Add(solver.Sum(y_train.values()) == 1)

        # Gauge compatibility
        for i, tck in enumerate(TRACK_CHOICES):
            for k, trn in enumerate(TRAIN_CHOICES):
                if tck.gauge != trn.gauge:
                    # If both binaries were 1 this would violate; enforce ≤1
                    solver.Add(x_track[i] + y_train[k] <= 1)

        # Capacity constraint
        capacity = solver.Sum(
            y_train[k] * TRAIN_CHOICES[k].capacity for k in y_train
        ) * 0.70 * (60 / 30) * 18 * n_sets  # seats/day with default params
        solver.Add(capacity >= demand_ppd)

        # Budget constraint
        capex = (
            solver.Sum(x_track[i] * TRACK_CHOICES[i].capex_per_km_eur for i in x_track)
            * km_route
            + solver.Sum(
                y_train[k] * TRAIN_CHOICES[k].purchase_cost_eur for k in y_train
            )
            * n_sets
        )
        solver.Add(capex <= budget_eur)

        # Objective: minimise CAPEX
        solver.Minimize(capex)

        result = solver.Solve()
        if result != _or_tools.Solver.OPTIMAL:
            logger.warning("MIP infeasible – reverting to greedy fallback")
            return _greedy(demand_ppd, budget_eur, graph, km_route)

        # Extract solution
        track = next(TRACK_CHOICES[i] for i, var in x_track.items() if var.solution_value() > 0.5)
        train = next(TRAIN_CHOICES[k] for k, var in y_train.items() if var.solution_value() > 0.5)
        fleet = int(n_sets.solution_value())
        cost = solver.Objective().Value()

        logger.info(
            "MIP picked %s + %s ×%d (CAPEX €%.1f M)",
            track.name,
            train.name,
            fleet,
            cost / 1e6,
        )
        return NetworkDesign(graph, track, train, demand_ppd, cost)

    # ------------------------------------------------------------------ Fallback
    return _greedy(demand_ppd, budget_eur, graph, km_route)


# ---------------------------------------------------------------------------#
# Greedy / compatibility-aware fallback                                      #
# ---------------------------------------------------------------------------#


def _greedy(
    demand_ppd: float,
    budget_eur: float,
    graph: nx.Graph,
    km_route: float,
) -> NetworkDesign:
    """
    Simple enumeration fallback when OR-Tools isn’t available or MIP fails.

    Chooses the cheapest compatible (track, train) pair whose cost ≤ budget.
    Fleet size is the minimum integer that meets demand.
    """
    best = None

    for track in TRACK_CHOICES:
        for train in TRAIN_CHOICES:
            if track.gauge != train.gauge:
                continue

            fleet = _fleet_size(demand_ppd, train)
            cost = track.capex_per_km_eur * km_route + train.purchase_cost_eur * fleet

            if cost <= budget_eur and (best is None or cost < best[2]):
                best = (track, train, cost)

    if best is None:
        logger.warning("Budget insufficient – returning empty network")
        return NetworkDesign(nx.Graph(), STD_TRACK, EMU, demand_ppd, 0.0)

    track, train, cost = best
    fleet = _fleet_size(demand_ppd, train)
    logger.info(
        "Greedy picked %s + %s ×%d (CAPEX €%.1f M)",
        track.name,
        train.name,
        fleet,
        cost / 1e6,
    )
    return NetworkDesign(graph, track, train, demand_ppd, cost)
