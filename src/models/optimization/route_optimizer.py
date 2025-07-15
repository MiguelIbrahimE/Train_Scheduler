import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import networkx as nx
from typing import List, Tuple, Dict, Optional
import logging
import yaml
from pathlib import Path
import heapq
from dataclasses import dataclass

@dataclass
class RouteNode:
    """Represents a node in the route optimization graph"""
    id: str
    lat: float
    lon: float
    elevation: float = 0.0
    cost: float = float('inf')
    parent: Optional['RouteNode'] = None
    
    def __lt__(self, other):
        return self.cost < other.cost

class RouteOptimizer:
    """Optimize railway routes using A* algorithm with terrain and demand considerations"""
    
    def __init__(self, config_path: str = "config/countries_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.terrain_costs = self.config['terrain_costs']
        self.logger = logging.getLogger(__name__)
        
        # Cost parameters
        self.base_cost_per_km = 10.0  # millions
        self.elevation_penalty = 0.001  # per meter
        self.gradient_penalty = 0.5  # per % gradient
        self.urban_penalty = 0.8  # multiplier for urban areas
        self.water_penalty = 2.0  # multiplier for water crossings
        
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points in kilometers"""
        
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def calculate_terrain_cost(self, point1: Tuple[float, float], point2: Tuple[float, float], 
                             elevation1: float, elevation2: float) -> float:
        """Calculate terrain-based cost between two points"""
        
        distance = self.calculate_distance(point1, point2)
        
        # Base cost
        base_cost = self.base_cost_per_km * distance
        
        # Elevation cost
        avg_elevation = (elevation1 + elevation2) / 2
        elevation_cost = avg_elevation * self.elevation_penalty * distance
        
        # Gradient cost
        elevation_diff = abs(elevation2 - elevation1)
        gradient = elevation_diff / (distance * 1000)  # gradient in decimal
        gradient_cost = gradient * self.gradient_penalty * distance
        
        return base_cost + elevation_cost + gradient_cost
    
    def calculate_demand_benefit(self, city1: Dict, city2: Dict) -> float:
        """Calculate demand benefit using gravity model"""
        
        pop1 = city1['population']
        pop2 = city2['population']
        distance = self.calculate_distance(
            (city1['lat'], city1['lon']), 
            (city2['lat'], city2['lon'])
        )
        
        # Gravity model: demand = (pop1 * pop2) / distance^2
        if distance > 0:
            demand = (pop1 * pop2) / (distance ** 2)
            return np.log(demand + 1)  # Log scale to avoid extreme values
        
        return 0
    
    def create_route_graph(self, cities: pd.DataFrame, existing_network: gpd.GeoDataFrame = None) -> nx.Graph:
        """Create a graph for route optimization"""
        
        G = nx.Graph()
        
        # Add city nodes
        for idx, city in cities.iterrows():
            G.add_node(
                city['city'],
                lat=city['lat'],
                lon=city['lon'],
                population=city['population'],
                node_type='city'
            )
        
        # Add existing railway nodes if available
        if existing_network is not None and not existing_network.empty:
            for idx, segment in existing_network.iterrows():
                coords = list(segment.geometry.coords)
                for i, (lon, lat) in enumerate(coords):
                    node_id = f"rail_{idx}_{i}"
                    G.add_node(
                        node_id,
                        lat=lat,
                        lon=lon,
                        node_type='railway'
                    )
        
        # Add edges between all city pairs
        city_names = cities['city'].tolist()
        for i, city1 in enumerate(city_names):
            for j, city2 in enumerate(city_names[i+1:], i+1):
                
                city1_data = cities[cities['city'] == city1].iloc[0]
                city2_data = cities[cities['city'] == city2].iloc[0]
                
                # Calculate costs
                terrain_cost = self.calculate_terrain_cost(
                    (city1_data['lat'], city1_data['lon']),
                    (city2_data['lat'], city2_data['lon']),
                    0, 0  # Elevation data would be fetched separately
                )
                
                demand_benefit = self.calculate_demand_benefit(
                    city1_data.to_dict(),
                    city2_data.to_dict()
                )
                
                # Combined cost (lower is better)
                combined_cost = terrain_cost - demand_benefit
                
                G.add_edge(
                    city1, city2,
                    weight=combined_cost,
                    distance=self.calculate_distance(
                        (city1_data['lat'], city1_data['lon']),
                        (city2_data['lat'], city2_data['lon'])
                    ),
                    terrain_cost=terrain_cost,
                    demand_benefit=demand_benefit
                )
        
        return G
    
    def astar_route_optimization(self, graph: nx.Graph, start: str, end: str, 
                                intermediate_cities: List[str] = None) -> Dict:
        """Find optimal route using A* algorithm"""
        
        if intermediate_cities is None:
            intermediate_cities = []
        
        # If no intermediate cities, use simple shortest path
        if not intermediate_cities:
            try:
                path = nx.shortest_path(graph, start, end, weight='weight')
                total_cost = nx.shortest_path_length(graph, start, end, weight='weight')
                
                return {
                    'path': path,
                    'total_cost': total_cost,
                    'segments': self._get_path_segments(graph, path)
                }
            except nx.NetworkXNoPath:
                return {'path': [], 'total_cost': float('inf'), 'segments': []}
        
        # For intermediate cities, solve TSP-like problem
        all_cities = [start] + intermediate_cities + [end]
        best_route = None
        best_cost = float('inf')
        
        # Try different permutations of intermediate cities
        from itertools import permutations
        
        for perm in permutations(intermediate_cities):
            current_route = [start] + list(perm) + [end]
            total_cost = 0
            valid_route = True
            
            # Calculate total cost for this permutation
            for i in range(len(current_route) - 1):
                try:
                    segment_cost = nx.shortest_path_length(
                        graph, current_route[i], current_route[i+1], weight='weight'
                    )
                    total_cost += segment_cost
                except nx.NetworkXNoPath:
                    valid_route = False
                    break
            
            if valid_route and total_cost < best_cost:
                best_cost = total_cost
                best_route = current_route
        
        if best_route:
            # Get full path with intermediate points
            full_path = []
            segments = []
            
            for i in range(len(best_route) - 1):
                segment_path = nx.shortest_path(
                    graph, best_route[i], best_route[i+1], weight='weight'
                )
                
                if i > 0:
                    segment_path = segment_path[1:]  # Remove duplicate node
                
                full_path.extend(segment_path)
                segments.append(self._get_path_segments(graph, segment_path))
            
            return {
                'path': full_path,
                'total_cost': best_cost,
                'segments': segments,
                'route_cities': best_route
            }
        
        return {'path': [], 'total_cost': float('inf'), 'segments': []}
    
    def _get_path_segments(self, graph: nx.Graph, path: List[str]) -> List[Dict]:
        """Get detailed information about path segments"""
        
        segments = []
        
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            
            edge_data = graph.edges[node1, node2]
            node1_data = graph.nodes[node1]
            node2_data = graph.nodes[node2]
            
            segment = {
                'from': node1,
                'to': node2,
                'from_coords': (node1_data['lat'], node1_data['lon']),
                'to_coords': (node2_data['lat'], node2_data['lon']),
                'distance_km': edge_data['distance'],
                'terrain_cost': edge_data['terrain_cost'],
                'demand_benefit': edge_data['demand_benefit'],
                'total_cost': edge_data['weight']
            }
            
            segments.append(segment)
        
        return segments
    
    def optimize_multiple_routes(self, cities: pd.DataFrame, route_pairs: List[Tuple[str, str]], 
                                existing_network: gpd.GeoDataFrame = None) -> Dict[str, Dict]:
        """Optimize multiple routes simultaneously"""
        
        # Create graph
        graph = self.create_route_graph(cities, existing_network)
        
        results = {}
        
        for i, (start, end) in enumerate(route_pairs):
            route_id = f"route_{i}_{start}_{end}"
            
            if start not in graph.nodes or end not in graph.nodes:
                self.logger.warning(f"Cities {start} or {end} not found in graph")
                continue
            
            # Find optimal route
            route_result = self.astar_route_optimization(graph, start, end)
            
            if route_result['path']:
                results[route_id] = route_result
                self.logger.info(f"✅ Optimized route {start} → {end}: {route_result['total_cost']:.2f} cost")
            else:
                self.logger.warning(f"❌ No path found for {start} → {end}")
        
        return results
    
    def learn_from_existing_network(self, existing_network: gpd.GeoDataFrame, 
                                   cities: pd.DataFrame) -> Dict[str, float]:
        """Learn cost parameters from existing railway network"""
        
        cost_patterns = {
            'terrain_preference': 0.0,
            'distance_efficiency': 0.0,
            'city_connectivity': 0.0,
            'elevation_avoidance': 0.0
        }
        
        if existing_network.empty:
            return cost_patterns
        
        # Analyze existing network characteristics
        total_length = existing_network.geometry.length.sum()
        avg_segment_length = existing_network.geometry.length.mean()
        
        # Calculate connectivity to cities
        city_connections = 0
        for _, city in cities.iterrows():
            city_point = Point(city['lon'], city['lat'])
            
            # Check if city is near any railway line
            for _, segment in existing_network.iterrows():
                if segment.geometry.distance(city_point) < 0.01:  # ~1km buffer
                    city_connections += 1
                    break
        
        connectivity_ratio = city_connections / len(cities)
        
        # Update cost patterns based on analysis
        cost_patterns['distance_efficiency'] = 1.0 / avg_segment_length if avg_segment_length > 0 else 0
        cost_patterns['city_connectivity'] = connectivity_ratio
        
        self.logger.info(f"Learned patterns: {cost_patterns}")
        
        return cost_patterns
    
    def export_routes_to_geojson(self, routes: Dict[str, Dict], cities: pd.DataFrame, 
                                output_path: str) -> None:
        """Export optimized routes to GeoJSON format"""
        
        features = []
        
        for route_id, route_data in routes.items():
            if not route_data['path']:
                continue
            
            # Create LineString from path coordinates
            coordinates = []
            for city_name in route_data['path']:
                city_data = cities[cities['city'] == city_name]
                if not city_data.empty:
                    coordinates.append([city_data.iloc[0]['lon'], city_data.iloc[0]['lat']])
            
            if len(coordinates) >= 2:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "route_id": route_id,
                        "total_cost": route_data['total_cost'],
                        "path": route_data['path'],
                        "total_distance": sum(seg['distance_km'] for seg in route_data['segments']),
                        "segment_count": len(route_data['segments'])
                    }
                }
                features.append(feature)
        
        # Create GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        self.logger.info(f"Exported {len(features)} routes to {output_path}")
    
    def generate_optimization_report(self, routes: Dict[str, Dict], cities: pd.DataFrame, 
                                   country: str) -> pd.DataFrame:
        """Generate detailed optimization report"""
        
        report_data = []
        
        for route_id, route_data in routes.items():
            if not route_data['path']:
                continue
            
            # Basic route info
            start_city = route_data['path'][0]
            end_city = route_data['path'][-1]
            total_distance = sum(seg['distance_km'] for seg in route_data['segments'])
            
            # Population served
            population_served = 0
            for city_name in route_data['path']:
                city_data = cities[cities['city'] == city_name]
                if not city_data.empty:
                    population_served += city_data.iloc[0]['population']
            
            # Cost analysis
            total_terrain_cost = sum(seg['terrain_cost'] for seg in route_data['segments'])
            total_demand_benefit = sum(seg['demand_benefit'] for seg in route_data['segments'])
            
            # Efficiency metrics
            direct_distance = self.calculate_distance(
                (cities[cities['city'] == start_city].iloc[0]['lat'], 
                 cities[cities['city'] == start_city].iloc[0]['lon']),
                (cities[cities['city'] == end_city].iloc[0]['lat'], 
                 cities[cities['city'] == end_city].iloc[0]['lon'])
            )
            
            route_efficiency = direct_distance / total_distance if total_distance > 0 else 0
            
            report_data.append({
                'route_id': route_id,
                'country': country,
                'start_city': start_city,
                'end_city': end_city,
                'intermediate_cities': len(route_data['path']) - 2,
                'total_distance_km': total_distance,
                'direct_distance_km': direct_distance,
                'route_efficiency': route_efficiency,
                'total_cost': route_data['total_cost'],
                'terrain_cost': total_terrain_cost,
                'demand_benefit': total_demand_benefit,
                'population_served': population_served,
                'cost_per_km': route_data['total_cost'] / total_distance if total_distance > 0 else 0,
                'population_per_km': population_served / total_distance if total_distance > 0 else 0
            })
        
        return pd.DataFrame(report_data)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    from src.data_processing.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    
    # Test with Germany
    country = 'germany'
    country_data = loader.load_all_country_data(country)
    
    if 'cities' in country_data:
        cities = country_data['cities']
        existing_network = country_data.get('railway_network', gpd.GeoDataFrame())
        
        # Create optimizer
        optimizer = RouteOptimizer()
        
        # Define some route pairs
        route_pairs = [
            ('Berlin', 'Munich'),
            ('Hamburg', 'Frankfurt'),
            ('Cologne', 'Stuttgart'),
            ('Berlin', 'Hamburg')
        ]
        
        # Optimize routes
        optimized_routes = optimizer.optimize_multiple_routes(
            cities, route_pairs, existing_network
        )
        
        # Generate report
        report = optimizer.generate_optimization_report(optimized_routes, cities, country)
        
        # Export results
        optimizer.export_routes_to_geojson(
            optimized_routes, cities, 
            f"data/output/routes/{country}_optimized_routes.geojson"
        )
        
        report.to_csv(f"data/output/reports/{country}_optimization_report.csv", index=False)
        
        print(f"\n=== Route Optimization Results for {country} ===")
        print(report.to_string(index=False))
    
    else:
        print(f"No city data available for {country}")