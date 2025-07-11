"""
intercity_planner.py - Smart intercity rail network planner
Uses graph theory and optimization to connect cities efficiently
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
from src.models import CityNode

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from geopy.distance import geodesic
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from src.models import CityNode
from .scenario_io import load_scenario, save_geojson
from .models import TrackType, TrainType, Gauge
from .cost import estimate_cost
from .terrain import load_dem
from .routing import trace_route
from . import OUTPUT_DIR
from dataclasses import dataclass, field
from shapely.geometry import LineString

from dataclasses import dataclass
from dotenv import load_dotenv
import os
load_dotenv()
log = logging.getLogger("bcpc.intercity")


# Hardcoded coordinates for Lebanese cities (since they're not in ScenarioRow)
CITY_COORDINATES = {
    'LB-BEY': {'name': 'Beirut', 'lat': 33.8938, 'lon': 35.5018},
    'LB-TRP': {'name': 'Tripoli', 'lat': 34.4367, 'lon': 35.8497},
    'LB-SID': {'name': 'Sidon', 'lat': 33.5606, 'lon': 35.3758},
    'LB-TYR': {'name': 'Tyre', 'lat': 33.2700, 'lon': 35.2033},
    'LB-BAA': {'name': 'Baalbek', 'lat': 34.0058, 'lon': 36.2181},
    'LB-BBS': {'name': 'Byblos', 'lat': 34.1230, 'lon': 35.6519},
    'LB-ALT': {'name': 'Aley', 'lat': 33.8106, 'lon': 35.5972}
}

@dataclass(frozen=True, eq=True)
class RailConnection:
    city1: str
    city2: str
    distance_km: float
    elevation_change: float
    cost_millions: float
    is_tunnel: bool
    geometry: LineString = field(compare=False, hash=False)  # exclude from hash/eq
    priority_score: float = 0


class IntercityNetworkPlanner:
    """Smart intercity rail network planner"""
    
    def __init__(self, cities: List[CityNode], budget_millions: float = 5000):
        self.cities = {city.code: city for city in cities}
        self.budget = budget_millions
        self.graph = nx.Graph()
        self.connections: List[RailConnection] = []
        
        # Initialize graph with cities
        for city in cities:
            self.graph.add_node(city.code, **city.__dict__)
    
    def get_elevation_at_point(self, lat: float, lon: float) -> float:
        """Get elevation at a point - simplified for now"""
        # Known elevations for Lebanese cities
        city_elevations = {
            'Beirut': 0,      # Sea level
            'Tripoli': 5,     # Near sea
            'Sidon': 0,       # Sea level
            'Tyre': 0,        # Sea level
            'Baalbek': 1170,  # High elevation
            'Byblos': 10,     # Near sea
            'Aley': 850       # Mountain town
        }
        
        # Find closest city and use its elevation
        for city_code, city in self.cities.items():
            if abs(city.lat - lat) < 0.1 and abs(city.lon - lon) < 0.1:
                city_name = city.name
                return city_elevations.get(city_name, 0)
        
        return 0  # Default to sea level
    
    def calculate_city_importance(self) -> None:
        """Calculate importance scores for cities based on multiple factors"""
        populations = [city.population for city in self.cities.values()]
        max_pop = max(populations)
        
        for city in self.cities.values():
            # Get elevation for the city
            city.elevation = self.get_elevation_at_point(city.lat, city.lon)
            
            # Factors: population, geographic centrality, economic importance
            pop_score = city.population / max_pop
            
            # Geographic centrality (how central is this city to others)
            distances = []
            for other in self.cities.values():
                if other.code != city.code:
                    dist = geodesic((city.lat, city.lon), (other.lat, other.lon)).km
                    distances.append(dist)
            centrality_score = 1 / (1 + np.mean(distances) / 100)
            
            # Economic hubs get bonus (capitals, ports)
            economic_bonus = 0.2 if city.name in ['Beirut', 'Tripoli', 'Sidon'] else 0
            
            city.importance_score = (pop_score * 0.5 + centrality_score * 0.3 + economic_bonus * 0.2)
    
    def calculate_connection_cost(self, city1: CityNode, city2: CityNode) -> Tuple[float, bool]:
        """Calculate cost of connecting two cities considering terrain"""
        distance = geodesic((city1.lat, city1.lon), (city2.lat, city2.lon)).km
        
        # Base cost per km (millions EUR)
        base_cost_per_km = 15
        
        # Elevation factor
        elevation_diff = abs(city1.elevation - city2.elevation)
        terrain_multiplier = 1 + (elevation_diff / 1000) * 2  # Double cost per 1000m elevation
        
        # Mountain detection
        is_mountainous = elevation_diff > 500 or max(city1.elevation, city2.elevation) > 800
        
        # Tunnel decision
        needs_tunnel = is_mountainous and elevation_diff > 300
        if needs_tunnel:
            terrain_multiplier *= 3  # Tunnels are expensive
        
        total_cost = distance * base_cost_per_km * terrain_multiplier
        
        return total_cost, needs_tunnel
    
    def generate_all_possible_connections(self) -> List[RailConnection]:
        connections = []
        gdf_cities = gpd.GeoDataFrame(
            [{'geometry': Point(c.lon, c.lat)} for c in self.cities.values()],
            crs="EPSG:4326"
        )
        dem = load_dem(gdf_cities)

        for city1, city2 in combinations(self.cities.values(), 2):
            origin = (city1.lon, city1.lat)
            dest = (city2.lon, city2.lat)

            try:
                geometry = trace_route(origin, dest, gdf_cities, dem)
            except Exception as e:
                log.warning(f"Routing failed for {city1.name}–{city2.name}: {e}")
                continue

            # Distance from geometry
            distance_km = geometry.length * 111  # approx deg to km

            # Elevation profile
            elev_start = dem.read(1)[dem.index(city1.lon, city1.lat)]
            elev_end   = dem.read(1)[dem.index(city2.lon, city2.lat)]
            elevation_change = abs(elev_start - elev_end)

            # Tunnel check
            is_mountainous = elevation_change > 500
            is_tunnel = is_mountainous

            # Cost estimation
            base_cost_per_km = 15
            terrain_multiplier = 1 + (elevation_change / 1000) * 2
            if is_tunnel:
                terrain_multiplier *= 3

            cost = distance_km * base_cost_per_km * terrain_multiplier

            # Priority score
            population_factor = (city1.population + city2.population) / 1_000_000
            distance_factor = 1 / (1 + distance_km / 100)
            importance_factor = (city1.importance_score + city2.importance_score) / 2
            priority = population_factor * 0.4 + distance_factor * 0.3 + importance_factor * 0.3

            connection = RailConnection(
                city1=city1.code,
                city2=city2.code,
                distance_km=distance_km,
                elevation_change=elevation_change,
                cost_millions=cost,
                is_tunnel=is_tunnel,
                geometry=geometry,
                priority_score=priority
            )
            connections.append(connection)

        return connections

    
    def optimize_network_mst(self) -> List[RailConnection]:
        """Use Minimum Spanning Tree algorithm for basic connectivity"""
        n = len(self.cities)
        city_codes = list(self.cities.keys())
        city_idx = {code: i for i, code in enumerate(city_codes)}
        
        # Create distance matrix
        dist_matrix = np.full((n, n), np.inf)
        connection_map = {}
        
        for conn in self.connections:
            i = city_idx[conn.city1]
            j = city_idx[conn.city2]
            # Use cost-weighted distance
            weight = conn.distance_km * (1 + conn.cost_millions / 1000)
            dist_matrix[i, j] = weight
            dist_matrix[j, i] = weight
            connection_map[(min(i, j), max(i, j))] = conn
        
        # Convert to sparse matrix and find MST
        sparse_matrix = csr_matrix(dist_matrix)
        mst = minimum_spanning_tree(sparse_matrix)
        
        # Extract connections from MST
        mst_connections = []
        mst_coo = mst.tocoo()
        
        for i, j in zip(mst_coo.row, mst_coo.col):
            key = (min(i, j), max(i, j))
            if key in connection_map:
                mst_connections.append(connection_map[key])
        
        return mst_connections
    
    def optimize_network_smart(self, max_connections: Optional[int] = None) -> List[RailConnection]:
        """Smart optimization considering budget, importance, and network effects"""
        # Start with MST for basic connectivity
        mst_connections = self.optimize_network_mst()
        selected = set(mst_connections)
        remaining_budget = self.budget - sum(c.cost_millions for c in selected)
        
        # Add high-priority connections within budget
        remaining_connections = [c for c in self.connections if c not in selected]
        remaining_connections.sort(key=lambda x: x.priority_score, reverse=True)
        
        for conn in remaining_connections:
            if remaining_budget >= conn.cost_millions:
                selected.add(conn)
                remaining_budget -= conn.cost_millions
                
                if max_connections and len(selected) >= max_connections:
                    break
        
        return list(selected)
    
    def plan_network(self) -> Dict:
        """Main planning method"""
        log.info("Starting intercity network planning")
        
        # Step 1: Calculate city importance
        self.calculate_city_importance()
        
        # Step 2: Generate all possible connections
        self.connections = self.generate_all_possible_connections()
        log.info(f"Generated {len(self.connections)} possible connections")
        
        # Step 3: Optimize network
        selected_connections = self.optimize_network_smart()
        log.info(f"Selected {len(selected_connections)} connections within budget")
        
        # Step 4: Build final network
        for conn in selected_connections:
            self.graph.add_edge(
                conn.city1, 
                conn.city2,
                connection=conn,
                weight=conn.distance_km
            )
        
        # Calculate network statistics
        stats = self.calculate_network_stats(selected_connections)
        
        return {
            'connections': selected_connections,
            'cities': self.cities,
            'stats': stats
        }
    
    def calculate_network_stats(self, connections: List[RailConnection]) -> Dict:
        """Calculate network statistics"""
        total_length = sum(c.distance_km for c in connections)
        total_cost = sum(c.cost_millions for c in connections)
        tunnel_length = sum(c.distance_km for c in connections if c.is_tunnel)
        
        # Network connectivity metrics
        temp_graph = nx.Graph()
        for conn in connections:
            temp_graph.add_edge(conn.city1, conn.city2, weight=conn.distance_km)
        
        if nx.is_connected(temp_graph):
            avg_path_length = nx.average_shortest_path_length(temp_graph, weight='weight')
            diameter = nx.diameter(temp_graph)
        else:
            avg_path_length = float('inf')
            diameter = float('inf')
        
        return {
            'total_length_km': total_length,
            'total_cost_millions': total_cost,
            'tunnel_length_km': tunnel_length,
            'num_connections': len(connections),
            'connected_cities': len([n for n in temp_graph.nodes if temp_graph.degree(n) > 0]),
            'avg_path_length': avg_path_length,
            'network_diameter': diameter,
            'cost_per_km': total_cost / total_length if total_length > 0 else 0
        }


def plan_intercity_network(csv_path: str, output_path: str, 
                          pop_threshold: int = 50000,
                          budget_millions: float = 5000) -> None:
    """Main entry point for intercity planning"""
    log.info(f"Loading cities from {csv_path}")
    
    # Load cities
    scenarios = load_scenario(Path(csv_path))
    
    # Convert to CityNodes using hardcoded coordinates
    cities = []
    for scenario in scenarios:
        if scenario.population >= pop_threshold:
            # Get coordinates from our lookup table
            if scenario.city_id in CITY_COORDINATES:
                coords = CITY_COORDINATES[scenario.city_id]
                city = CityNode(
                    name=scenario.city_name,
                    code=scenario.city_id,
                    lat=coords['lat'],
                    lon=coords['lon'],
                    population=scenario.population,
                    elevation=0  # Will be filled by calculate_city_importance
                )
                cities.append(city)
            else:
                log.warning(f"No coordinates found for {scenario.city_name} ({scenario.city_id})")
    
    log.info(f"Planning network for {len(cities)} cities with budget €{budget_millions}M")
    
    # Create planner and plan network
    planner = IntercityNetworkPlanner(cities, budget_millions)
    result = planner.plan_network()
    # Decide track type based on terrain or tunnel flag
    if conn.is_tunnel:
        track_type = "tunnel"
    elif conn.elevation_change > 200:
        track_type = "elevated"
    else:
        track_type = "ground"
    # Convert to GeoDataFrame
    connections_data = []
    for conn in result['connections']:
        connections_data.append({
            'geometry': conn.geometry,
            'city1': conn.city1,
            'city2': conn.city2,
            'distance_km': conn.distance_km,
            'cost_millions': conn.cost_millions,
            'is_tunnel': conn.is_tunnel,
            'priority': conn.priority_score,
            'elevation_change': conn.elevation_change
        })

    
    gdf = gpd.GeoDataFrame(connections_data, crs='EPSG:4326')
    
    # Save outputs
    output_path = Path(output_path)
    save_geojson(gdf, output_path)
    
    # Save statistics
    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(result['stats'], f, indent=2)
    
    log.info(f"Network saved to {output_path}")
    log.info(f"Total cost: €{result['stats']['total_cost_millions']:.1f}M")
    log.info(f"Total length: {result['stats']['total_length_km']:.1f}km")


__all__ = ['plan_intercity_network', 'IntercityNetworkPlanner']