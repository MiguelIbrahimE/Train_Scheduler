# Railway Network Planning AI System
# Complete implementation for learning from efficient rail networks and planning new ones

import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import folium
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import networkx as nx
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import elevation
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
import heapq
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class City:
    """Represents a city with all relevant data"""
    id: str
    name: str
    country: str
    lat: float
    lon: float
    population: int
    area_km2: float
    elevation: float
    tourism_index: float
    economic_index: float
    students: int
    workers: int

@dataclass
class RouteSegment:
    """Represents a segment of railway route"""
    start_city: str
    end_city: str
    distance_km: float
    terrain_difficulty: float
    cost_per_km: float
    terrain_type: str  # 'flat', 'hilly', 'mountainous', 'water_crossing'
    underground_percentage: float

@dataclass
class TrainSpecs:
    """Train specifications"""
    name: str
    capacity: int
    max_speed: int
    cost: float
    double_decker: bool
    high_speed: bool

class DataCollector:
    """Collects data about cities from various online sources"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="railway_planner")
        self.cache_db = "city_cache.db"
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        conn = sqlite3.connect(self.cache_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS city_data (
                city_id TEXT PRIMARY KEY,
                data TEXT,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def get_city_data(self, city_id: str, city_name: str, use_cache: bool = True) -> City:
        """Get comprehensive city data"""
        if use_cache:
            cached_data = self._get_from_cache(city_id)
            if cached_data:
                return cached_data
        
        logger.info(f"Fetching data for {city_name}")
        
        # Get coordinates
        location = self.geolocator.geocode(city_name)
        if not location:
            raise ValueError(f"Could not find coordinates for {city_name}")
        
        lat, lon = location.latitude, location.longitude
        
        # Get additional data (simulated with realistic values)
        city_data = self._fetch_city_details(city_name, lat, lon)
        
        city = City(
            id=city_id,
            name=city_name,
            country=city_data['country'],
            lat=lat,
            lon=lon,
            population=city_data['population'],
            area_km2=city_data['area'],
            elevation=city_data['elevation'],
            tourism_index=city_data['tourism_index'],
            economic_index=city_data['economic_index'],
            students=city_data['students'],
            workers=city_data['workers']
        )
        
        self._save_to_cache(city_id, city)
        return city
    
    def _fetch_city_details(self, city_name: str, lat: float, lon: float) -> Dict:
        """Fetch detailed city information (simulated)"""
        # In real implementation, this would call multiple APIs:
        # - WorldBank for population/economic data
        # - Tourism APIs for tourism index
        # - Elevation APIs for terrain data
        # - Census APIs for student/worker data
        
        # Simulated realistic data
        base_pop = np.random.randint(50000, 2000000)
        return {
            'country': self._get_country_from_coords(lat, lon),
            'population': base_pop,
            'area': np.random.uniform(50, 500),
            'elevation': self._get_elevation(lat, lon),
            'tourism_index': np.random.uniform(0.1, 1.0),
            'economic_index': np.random.uniform(0.3, 1.0),
            'students': int(base_pop * np.random.uniform(0.1, 0.25)),
            'workers': int(base_pop * np.random.uniform(0.4, 0.7))
        }
    
    def _get_country_from_coords(self, lat: float, lon: float) -> str:
        """Get country from coordinates"""
        # Simplified country detection
        try:
            location = self.geolocator.reverse((lat, lon))
            return location.raw['address'].get('country', 'Unknown')
        except:
            return 'Unknown'
    
    def _get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation for coordinates"""
        # Simulated elevation data
        return np.random.uniform(0, 1000)
    
    def _get_from_cache(self, city_id: str) -> Optional[City]:
        """Get city data from cache"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.execute(
            'SELECT data FROM city_data WHERE city_id = ? AND timestamp > ?',
            (city_id, time.time() - 86400)  # 24 hour cache
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        return None
    
    def _save_to_cache(self, city_id: str, city: City):
        """Save city data to cache"""
        conn = sqlite3.connect(self.cache_db)
        conn.execute(
            'INSERT OR REPLACE INTO city_data (city_id, data, timestamp) VALUES (?, ?, ?)',
            (city_id, pickle.dumps(city), time.time())
        )
        conn.commit()
        conn.close()

class TerrainAnalyzer:
    """Analyzes terrain between cities for route planning"""
    
    def __init__(self):
        self.terrain_costs = {
            'flat': 1.0,
            'hilly': 1.3,
            'mountainous': 2.0,
            'water_crossing': 2.5,
            'urban': 1.8
        }
    
    def analyze_route_terrain(self, city1: City, city2: City, num_points: int = 50) -> Dict:
        """Analyze terrain along potential route"""
        points = self._get_route_points(city1, city2, num_points)
        elevations = [self._get_point_elevation(lat, lon) for lat, lon in points]
        
        # Calculate terrain metrics
        elevation_change = max(elevations) - min(elevations)
        avg_elevation = np.mean(elevations)
        elevation_variance = np.var(elevations)
        
        # Determine terrain type
        if elevation_change < 100:
            terrain_type = 'flat'
        elif elevation_change < 500:
            terrain_type = 'hilly' 
        else:
            terrain_type = 'mountainous'
        
        # Check for water crossings (simplified)
        has_water_crossing = self._check_water_crossing(city1, city2)
        if has_water_crossing:
            terrain_type = 'water_crossing'
        
        difficulty = self._calculate_terrain_difficulty(elevation_change, elevation_variance)
        
        return {
            'terrain_type': terrain_type,
            'difficulty': difficulty,
            'elevation_change': elevation_change,
            'avg_elevation': avg_elevation,
            'cost_multiplier': self.terrain_costs[terrain_type]
        }
    
    def _get_route_points(self, city1: City, city2: City, num_points: int) -> List[Tuple[float, float]]:
        """Generate points along straight line between cities"""
        lats = np.linspace(city1.lat, city2.lat, num_points)
        lons = np.linspace(city1.lon, city2.lon, num_points)
        return list(zip(lats, lons))
    
    def _get_point_elevation(self, lat: float, lon: float) -> float:
        """Get elevation for a point (simulated)"""
        return np.random.uniform(0, 1000)
    
    def _check_water_crossing(self, city1: City, city2: City) -> bool:
        """Check if route crosses major water body (simplified)"""
        # Simplified water crossing detection
        return np.random.random() < 0.1
    
    def _calculate_terrain_difficulty(self, elevation_change: float, variance: float) -> float:
        """Calculate terrain difficulty score (0-1)"""
        return min(1.0, (elevation_change / 1000 + variance / 50000) / 2)

class DemandCalculator:
    """Calculates passenger demand between cities"""
    
    def calculate_demand(self, city1: City, city2: City) -> Dict[str, int]:
        """Calculate passenger demand between two cities"""
        distance = geodesic((city1.lat, city1.lon), (city2.lat, city2.lon)).kilometers
        
        # Distance decay factor
        distance_factor = max(0.1, 1 / (1 + distance / 100))
        
        # Worker demand (commuting)
        worker_demand = self._calculate_worker_demand(city1, city2, distance_factor)
        
        # Student demand
        student_demand = self._calculate_student_demand(city1, city2, distance_factor)
        
        # Tourism demand
        tourism_demand = self._calculate_tourism_demand(city1, city2, distance_factor)
        
        total_daily_demand = worker_demand + student_demand + tourism_demand
        peak_hour_demand = int(total_daily_demand * 0.3)  # 30% in peak hour
        
        return {
            'worker_demand': worker_demand,
            'student_demand': student_demand,
            'tourism_demand': tourism_demand,
            'total_daily': total_daily_demand,
            'peak_hour': peak_hour_demand
        }
    
    def _calculate_worker_demand(self, city1: City, city2: City, distance_factor: float) -> int:
        """Calculate commuter demand"""
        # Simplified model: larger city attracts workers from smaller city
        if city1.population > city2.population:
            base_demand = int(city2.workers * 0.1 * city1.economic_index * distance_factor)
        else:
            base_demand = int(city1.workers * 0.1 * city2.economic_index * distance_factor)
        
        return max(0, base_demand)
    
    def _calculate_student_demand(self, city1: City, city2: City, distance_factor: float) -> int:
        """Calculate student travel demand"""
        avg_students = (city1.students + city2.students) / 2
        return int(avg_students * 0.05 * distance_factor)
    
    def _calculate_tourism_demand(self, city1: City, city2: City, distance_factor: float) -> int:
        """Calculate tourism demand"""
        avg_tourism = (city1.tourism_index + city2.tourism_index) / 2
        base_tourism = (city1.population + city2.population) / 2
        return int(base_tourism * avg_tourism * 0.02 * distance_factor)

class CostCalculator:
    """Calculates construction and operational costs"""
    
    def __init__(self):
        self.base_costs = {
            'rail_per_km': 5000000,  # €5M per km base cost
            'electrification_per_km': 1000000,  # €1M per km
            'station_small': 10000000,  # €10M
            'station_medium': 25000000,  # €25M
            'station_large': 50000000,  # €50M
            'tunnel_per_km': 50000000,  # €50M per km
            'bridge_per_km': 20000000,  # €20M per km
        }
        
        self.train_specs = {
            'regional': TrainSpecs('Regional DMU', 200, 120, 3000000, False, False),
            'intercity': TrainSpecs('Intercity EMU', 400, 160, 8000000, False, False),
            'high_speed': TrainSpecs('High Speed', 500, 300, 25000000, False, True),
            'double_decker': TrainSpecs('Double Decker', 800, 160, 12000000, True, False)
        }
    
    def calculate_route_cost(self, segment: RouteSegment, electrify: bool = True) -> Dict:
        """Calculate total cost for a route segment"""
        base_cost = self.base_costs['rail_per_km'] * segment.distance_km
        terrain_cost = base_cost * segment.terrain_difficulty * 0.5
        
        # Underground sections cost more
        underground_cost = (segment.underground_percentage / 100) * segment.distance_km * self.base_costs['tunnel_per_km']
        
        electrification_cost = 0
        if electrify:
            electrification_cost = self.base_costs['electrification_per_km'] * segment.distance_km
        
        total_cost = base_cost + terrain_cost + underground_cost + electrification_cost
        
        return {
            'base_cost': base_cost,
            'terrain_cost': terrain_cost,
            'underground_cost': underground_cost,
            'electrification_cost': electrification_cost,
            'total_cost': total_cost,
            'cost_per_km': total_cost / segment.distance_km
        }
    
    def recommend_train_type(self, demand: Dict, distance_km: float) -> str:
        """Recommend appropriate train type based on demand and distance"""
        peak_demand = demand['peak_hour']
        
        if distance_km > 300 and peak_demand > 300:
            return 'high_speed'
        elif peak_demand > 600:
            return 'double_decker'
        elif distance_km > 100:
            return 'intercity'
        else:
            return 'regional'
    
    def calculate_station_cost(self, city: City, demand: Dict) -> Dict:
        """Calculate station construction cost"""
        peak_demand = demand['peak_hour']
        
        if peak_demand < 500:
            station_type = 'small'
            platforms = 2
        elif peak_demand < 1500:
            station_type = 'medium'
            platforms = 4
        else:
            station_type = 'large'
            platforms = 6
        
        base_cost = self.base_costs[f'station_{station_type}']
        
        return {
            'station_type': station_type,
            'platforms': platforms,
            'cost': base_cost
        }

class RouteOptimizer:
    """Optimizes railway routes using pathfinding algorithms"""
    
    def __init__(self, cities: List[City], terrain_analyzer: TerrainAnalyzer):
        self.cities = cities
        self.terrain_analyzer = terrain_analyzer
        self.graph = self._build_graph()
    
    def _build_graph(self) -> nx.Graph:
        """Build graph representation of possible connections"""
        G = nx.Graph()
        
        # Add cities as nodes
        for city in self.cities:
            G.add_node(city.id, city=city)
        
        # Add edges between all city pairs (complete graph)
        for i, city1 in enumerate(self.cities):
            for city2 in self.cities[i+1:]:
                distance = geodesic((city1.lat, city1.lon), (city2.lat, city2.lon)).kilometers
                terrain = self.terrain_analyzer.analyze_route_terrain(city1, city2)
                
                # Weight combines distance and terrain difficulty
                weight = distance * (1 + terrain['difficulty'])
                
                G.add_edge(city1.id, city2.id, 
                          weight=weight,
                          distance=distance,
                          terrain=terrain)
        
        return G
    
    def find_optimal_routes(self, start_city_id: str, end_city_id: str, num_alternatives: int = 5) -> List[Dict]:
        """Find multiple optimal route alternatives"""
        routes = []
        
        # Find shortest path
        try:
            shortest_path = nx.shortest_path(self.graph, start_city_id, end_city_id, weight='weight')
            routes.append(self._analyze_path(shortest_path))
        except nx.NetworkXNoPath:
            logger.error(f"No path found between {start_city_id} and {end_city_id}")
            return []
        
        # Find alternative paths by temporarily removing edges
        for _ in range(num_alternatives - 1):
            # Remove some edges from the shortest path and find alternative
            temp_graph = self.graph.copy()
            if len(routes) > 0:
                last_path = routes[-1]['path']
                # Remove middle edges to force different route
                if len(last_path) > 2:
                    mid_idx = len(last_path) // 2
                    temp_graph.remove_edge(last_path[mid_idx-1], last_path[mid_idx])
            
            try:
                alt_path = nx.shortest_path(temp_graph, start_city_id, end_city_id, weight='weight')
                route_analysis = self._analyze_path(alt_path)
                if route_analysis not in routes:  # Avoid duplicates
                    routes.append(route_analysis)
            except nx.NetworkXNoPath:
                break
        
        return sorted(routes, key=lambda x: x['total_score'])
    
    def _analyze_path(self, path: List[str]) -> Dict:
        """Analyze a complete path"""
        total_distance = 0
        total_difficulty = 0
        segments = []
        
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i+1]]
            total_distance += edge_data['distance']
            total_difficulty += edge_data['terrain']['difficulty']
            segments.append({
                'from': path[i],
                'to': path[i+1],
                'distance': edge_data['distance'],
                'terrain': edge_data['terrain']
            })
        
        avg_difficulty = total_difficulty / len(segments) if segments else 0
        
        # Calculate composite score (lower is better)
        total_score = total_distance + (avg_difficulty * 100)
        
        return {
            'path': path,
            'segments': segments,
            'total_distance': total_distance,
            'avg_difficulty': avg_difficulty,
            'total_score': total_score
        }
    
    def optimize_network(self, city_pairs: List[Tuple[str, str]]) -> Dict:
        """Optimize entire network of connections"""
        network_routes = {}
        total_cost = 0
        
        for start_city, end_city in city_pairs:
            routes = self.find_optimal_routes(start_city, end_city)
            if routes:
                best_route = routes[0]
                network_routes[f"{start_city}-{end_city}"] = best_route
                
                # Estimate cost (simplified)
                cost_calc = CostCalculator()
                route_cost = cost_calc.base_costs['rail_per_km'] * best_route['total_distance']
                total_cost += route_cost
        
        return {
            'routes': network_routes,
            'total_cost': total_cost,
            'total_distance': sum(route['total_distance'] for route in network_routes.values())
        }

class RailwayML:
    """Machine Learning component to learn from existing efficient networks"""
    
    def __init__(self):
        self.models = {
            'cost_predictor': RandomForestRegressor(n_estimators=100),
            'demand_predictor': GradientBoostingRegressor(n_estimators=100),
            'efficiency_scorer': RandomForestRegressor(n_estimators=100)
        }
        self.scalers = {
            'cost': StandardScaler(),
            'demand': StandardScaler(),
            'efficiency': StandardScaler()
        }
        self.is_trained = False
    
    def prepare_training_data(self, efficient_networks: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from efficient railway networks"""
        features = []
        targets = []
        
        for network in efficient_networks:
            for route in network['routes']:
                # Feature engineering
                feature_vector = self._extract_features(route)
                features.append(feature_vector)
                
                # Target values (efficiency metrics)
                target_vector = [
                    route['cost_efficiency'],
                    route['passenger_satisfaction'],
                    route['environmental_score']
                ]
                targets.append(target_vector)
        
        return np.array(features), np.array(targets)
    
    def _extract_features(self, route: Dict) -> List[float]:
        """Extract features from route data"""
        return [
            route['distance'],
            route['population_served'],
            route['terrain_difficulty'],
            route['num_stations'],
            route['electrification_ratio'],
            route['average_speed'],
            route['frequency_per_hour'],
            route['cost_per_km'],
            route['elevation_change'],
            route['urban_percentage']
        ]
    
    def train_models(self, efficient_networks: List[Dict]):
        """Train ML models on efficient network data"""
        logger.info("Training ML models on efficient railway networks...")
        
        # Generate sample training data (in real implementation, this would be actual data)
        features, targets = self._generate_sample_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
        
        # Scale features
        X_train_scaled = self.scalers['efficiency'].fit_transform(X_train)
        X_test_scaled = self.scalers['efficiency'].transform(X_test)
        
        # Train efficiency scorer
        self.models['efficiency_scorer'].fit(X_train_scaled, y_train[:, 0])  # Use first target as efficiency
        
        # Evaluate
        score = self.models['efficiency_scorer'].score(X_test_scaled, y_test[:, 0])
        logger.info(f"Model training completed. Test score: {score:.3f}")
        
        self.is_trained = True
    
    def _generate_sample_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic sample training data"""
        n_samples = 1000
        
        # Sample features from realistic distributions
        features = np.random.rand(n_samples, 10)
        
        # Scale features to realistic ranges
        features[:, 0] *= 500  # distance (0-500 km)
        features[:, 1] *= 2000000  # population (0-2M)
        features[:, 2] *= 1  # terrain difficulty (0-1)
        features[:, 3] *= 20  # num stations (0-20)
        features[:, 4] *= 1  # electrification ratio (0-1)
        features[:, 5] = features[:, 5] * 200 + 50  # speed (50-250 km/h)
        features[:, 6] *= 10  # frequency (0-10 per hour)
        features[:, 7] = features[:, 7] * 10000000 + 1000000  # cost per km (1-11M)
        features[:, 8] *= 1000  # elevation change (0-1000m)
        features[:, 9] *= 1  # urban percentage (0-1)
        
        # Generate targets based on realistic relationships
        targets = np.zeros((n_samples, 3))
        
        # Efficiency score (higher is better)
        targets[:, 0] = (
            1 / (features[:, 7] / 1000000) +  # Lower cost = higher efficiency
            features[:, 5] / 100 +  # Higher speed = higher efficiency
            features[:, 6] * 0.1 +  # Higher frequency = higher efficiency
            (1 - features[:, 2]) * 0.5  # Lower terrain difficulty = higher efficiency
        )
        
        # Add noise
        targets += np.random.normal(0, 0.1, targets.shape)
        
        return features, targets
    
    def predict_route_efficiency(self, route_features: Dict) -> float:
        """Predict efficiency score for a proposed route"""
        if not self.is_trained:
            logger.warning("Model not trained yet. Training with sample data...")
            self.train_models([])
        
        feature_vector = np.array([self._extract_features_from_dict(route_features)]).reshape(1, -1)
        scaled_features = self.scalers['efficiency'].transform(feature_vector)
        
        efficiency_score = self.models['efficiency_scorer'].predict(scaled_features)[0]
        return max(0, min(10, efficiency_score))  # Clip to 0-10 range
    
    def _extract_features_from_dict(self, route_dict: Dict) -> List[float]:
        """Extract features from route dictionary"""
        return [
            route_dict.get('distance', 0),
            route_dict.get('population_served', 0),
            route_dict.get('terrain_difficulty', 0),
            route_dict.get('num_stations', 0),
            route_dict.get('electrification_ratio', 1.0),
            route_dict.get('average_speed', 120),
            route_dict.get('frequency_per_hour', 2),
            route_dict.get('cost_per_km', 5000000),
            route_dict.get('elevation_change', 0),
            route_dict.get('urban_percentage', 0.5)
        ]
    
    def recommend_improvements(self, route: Dict) -> List[str]:
        """Recommend improvements for a route based on learned patterns"""
        recommendations = []
        
        if route.get('terrain_difficulty', 0) > 0.7:
            recommendations.append("Consider more tunneling to reduce terrain impact")
        
        if route.get('average_speed', 0) < 100:
            recommendations.append("Electrification could increase average speed")
        
        if route.get('frequency_per_hour', 0) < 2:
            recommendations.append("Increase service frequency to improve attractiveness")
        
        return recommendations

class NetworkVisualizer:
    """Creates visualizations of railway networks"""
    
    def __init__(self):
        self.colors = {
            'underground': 'black',
            'above_ground': 'blue', 
            'on_ground': 'green',
            'high_speed': 'red'
        }
    
    def create_network_map(self, cities: List[City], routes: Dict, output_file: str = "railway_network.html"):
        """Create interactive map of railway network"""
        # Find center point for map
        center_lat = np.mean([city.lat for city in cities])
        center_lon = np.mean([city.lon for city in cities])
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
        
        # Add cities
        for city in cities:
            folium.CircleMarker(
                location=[city.lat, city.lon],
                radius=max(5, min(20, city.population / 100000)),
                popup=f"{city.name}<br>Population: {city.population:,}<br>Elevation: {city.elevation:.0f}m",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7
            ).add_to(m)
        
        # Add routes
        city_dict = {city.id: city for city in cities}
        
        for route_name, route_data in routes.items():
            if 'segments' in route_data:
                for segment in route_data['segments']:
                    start_city = city_dict[segment['from']]
                    end_city = city_dict[segment['to']]
                    
                    # Determine line color based on terrain
                    terrain_type = segment['terrain']['terrain_type']
                    if terrain_type == 'mountainous':
                        color = self.colors['underground']
                        weight = 4
                    elif terrain_type == 'flat':
                        color = self.colors['on_ground']
                        weight = 3
                    else:
                        color = self.colors['above_ground']
                        weight = 3
                    
                    folium.PolyLine(
                        locations=[[start_city.lat, start_city.lon], [end_city.lat, end_city.lon]],
                        color=color,
                        weight=weight,
                        opacity=0.8,
                        popup=f"Distance: {segment['distance']:.1f}km<br>Terrain: {terrain_type}"
                    ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <b>Railway Network</b><br>
        <i class="fa fa-circle" style="color:black"></i> Underground<br>
        <i class="fa fa-circle" style="color:blue"></i> Above Ground<br>
        <i class="fa fa-circle" style="color:green"></i> On Ground<br>
        <i class="fa fa-circle" style="color:red"></i> Cities<br>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(output_file)
        logger.info(f"Interactive map saved to {output_file}")
        
        return m
    
    def create_cost_analysis_chart(self, routes: Dict, output_file: str = "cost_analysis.png"):
        """Create cost analysis visualization"""
        route_names = list(routes.keys())
        costs = [route.get('total_cost', 0) / 1000000 for route in routes.values()]  # Convert to millions
        distances = [route.get('total_distance', 0) for route in routes.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost by route
        ax1.bar(range(len(route_names)), costs, color='skyblue')
        ax1.set_xlabel('Routes')
        ax1.set_ylabel('Cost (Million €)')
        ax1.set_title('Construction Cost by Route')
        ax1.set_xticks(range(len(route_names)))
        ax1.set_xticklabels(route_names, rotation=45, ha='right')
        
        # Cost per km
        cost_per_km = [c/d if d > 0 else 0 for c, d in zip(costs, distances)]
        ax2.bar(range(len(route_names)), cost_per_km, color='lightcoral')
        ax2.set_xlabel('Routes')
        ax2.set_ylabel('Cost per km (Million €/km)')
        ax2.set_title('Cost Efficiency by Route')
        ax2.set_xticks(range(len(route_names)))
        ax2.set_xticklabels(route_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cost analysis chart saved to {output_file}")

class RailwayPlannerSystem:
    """Main system that orchestrates all components"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.terrain_analyzer = TerrainAnalyzer()
        self.demand_calculator = DemandCalculator()
        self.cost_calculator = CostCalculator()
        self.ml_system = RailwayML()
        self.visualizer = NetworkVisualizer()
        self.cities = []
        self.route_optimizer = None
    
    def load_cities_from_csv(self, csv_file: str) -> List[City]:
        """Load cities from CSV file"""
        logger.info(f"Loading cities from {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            cities = []
            
            for _, row in df.iterrows():
                city_id = row['city_id']
                city_name = row['city_name']
                
                city = self.data_collector.get_city_data(city_id, city_name)
                cities.append(city)
                
                logger.info(f"Loaded data for {city_name}")
            
            self.cities = cities
            self.route_optimizer = RouteOptimizer(cities, self.terrain_analyzer)
            
            return cities
            
        except Exception as e:
            logger.error(f"Error loading cities from CSV: {e}")
            return []
    
    def analyze_network_demand(self) -> Dict:
        """Analyze passenger demand across the network"""
        demand_matrix = {}
        total_demand = 0
        
        for i, city1 in enumerate(self.cities):
            for city2 in self.cities[i+1:]:
                demand = self.demand_calculator.calculate_demand(city1, city2)
                route_key = f"{city1.id}-{city2.id}"
                demand_matrix[route_key] = demand
                total_demand += demand['total_daily']
        
        return {
            'demand_matrix': demand_matrix,
            'total_daily_demand': total_demand,
            'network_size': len(self.cities)
        }
    
    def plan_optimal_network(self, budget: float = 1000000000) -> Dict:  # €1B default budget
        """Plan optimal railway network within budget"""
        logger.info("Planning optimal railway network...")
        
        # Analyze demand
        demand_analysis = self.analyze_network_demand()
        
        # Generate city pairs sorted by demand
        city_pairs = []
        for route_key, demand in demand_analysis['demand_matrix'].items():
            city1_id, city2_id = route_key.split('-')
            city_pairs.append((city1_id, city2_id, demand['total_daily']))
        
        # Sort by demand (highest first)
        city_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Build network within budget
        selected_routes = {}
        total_cost = 0
        
        for city1_id, city2_id, demand in city_pairs:
            if total_cost >= budget:
                break
            
            # Find optimal route
            routes = self.route_optimizer.find_optimal_routes(city1_id, city2_id)
            if not routes:
                continue
            
            best_route = routes[0]
            
            # Calculate detailed costs
            route_cost = 0
            enhanced_segments = []
            
            for segment in best_route['segments']:
                # Create RouteSegment object
                route_segment = RouteSegment(
                    start_city=segment['from'],
                    end_city=segment['to'],
                    distance_km=segment['distance'],
                    terrain_difficulty=segment['terrain']['difficulty'],
                    cost_per_km=self.cost_calculator.base_costs['rail_per_km'],
                    terrain_type=segment['terrain']['terrain_type'],
                    underground_percentage=20 if segment['terrain']['terrain_type'] == 'mountainous' else 0
                )
                
                segment_cost = self.cost_calculator.calculate_route_cost(route_segment)
                route_cost += segment_cost['total_cost']
                
                enhanced_segments.append({
                    **segment,
                    'cost_analysis': segment_cost
                })
            
            # Check if we can afford this route
            if total_cost + route_cost <= budget:
                route_key = f"{city1_id}-{city2_id}"
                
                # Get demand data
                demand_data = demand_analysis['demand_matrix'][route_key]
                
                # ML prediction for efficiency
                route_features = {
                    'distance': best_route['total_distance'],
                    'population_served': sum(city.population for city in self.cities 
                                           if city.id in [city1_id, city2_id]),
                    'terrain_difficulty': best_route['avg_difficulty'],
                    'num_stations': len(best_route['path']),
                    'cost_per_km': route_cost / best_route['total_distance']
                }
                
                efficiency_score = self.ml_system.predict_route_efficiency(route_features)
                
                selected_routes[route_key] = {
                    **best_route,
                    'segments': enhanced_segments,
                    'total_cost': route_cost,
                    'demand': demand_data,
                    'efficiency_score': efficiency_score,
                    'train_recommendation': self.cost_calculator.recommend_train_type(
                        demand_data, best_route['total_distance']
                    )
                }
                
                total_cost += route_cost
                logger.info(f"Added route {route_key}: {route_cost/1000000:.1f}M€, "
                           f"Efficiency: {efficiency_score:.2f}")
        
        # Calculate network statistics
        total_distance = sum(route['total_distance'] for route in selected_routes.values())
        avg_efficiency = np.mean([route['efficiency_score'] for route in selected_routes.values()])
        
        network_plan = {
            'routes': selected_routes,
            'total_cost': total_cost,
            'budget_used_percent': (total_cost / budget) * 100,
            'total_distance': total_distance,
            'avg_efficiency_score': avg_efficiency,
            'num_routes': len(selected_routes),
            'cost_per_km': total_cost / total_distance if total_distance > 0 else 0
        }
        
        logger.info(f"Network planning completed:")
        logger.info(f"- {len(selected_routes)} routes planned")
        logger.info(f"- Total cost: €{total_cost/1000000:.1f}M ({(total_cost/budget)*100:.1f}% of budget)")
        logger.info(f"- Total distance: {total_distance:.1f} km")
        logger.info(f"- Average efficiency score: {avg_efficiency:.2f}")
        
        return network_plan
    
    def generate_reports(self, network_plan: Dict, output_dir: str = "output"):
        """Generate comprehensive reports and visualizations"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create interactive map
        map_file = f"{output_dir}/railway_network_map.html"
        self.visualizer.create_network_map(self.cities, network_plan['routes'], map_file)
        
        # Create cost analysis
        cost_chart_file = f"{output_dir}/cost_analysis.png"
        self.visualizer.create_cost_analysis_chart(network_plan['routes'], cost_chart_file)
        
        # Generate detailed report
        report_file = f"{output_dir}/network_analysis_report.txt"
        self._generate_text_report(network_plan, report_file)
        
        # Save network data as JSON
        json_file = f"{output_dir}/network_data.json"
        with open(json_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_plan = self._make_json_serializable(network_plan)
            json.dump(serializable_plan, f, indent=2)
        
        logger.info(f"Reports generated in {output_dir}/ directory")
        
        return {
            'map_file': map_file,
            'cost_chart': cost_chart_file,
            'report_file': report_file,
            'json_file': json_file
        }
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _generate_text_report(self, network_plan: Dict, output_file: str):
        """Generate detailed text report"""
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("RAILWAY NETWORK PLANNING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Routes Planned: {network_plan['num_routes']}\n")
            f.write(f"Total Construction Cost: €{network_plan['total_cost']/1000000:.1f}M\n")
            f.write(f"Total Network Distance: {network_plan['total_distance']:.1f} km\n")
            f.write(f"Average Cost per km: €{network_plan['cost_per_km']/1000000:.2f}M\n")
            f.write(f"Average Efficiency Score: {network_plan['avg_efficiency_score']:.2f}/10\n")
            f.write(f"Budget Utilization: {network_plan['budget_used_percent']:.1f}%\n\n")
            
            f.write("DETAILED ROUTE ANALYSIS\n")
            f.write("-" * 25 + "\n")
            
            for route_name, route_data in network_plan['routes'].items():
                f.write(f"\nRoute: {route_name}\n")
                f.write(f"  Distance: {route_data['total_distance']:.1f} km\n")
                f.write(f"  Cost: €{route_data['total_cost']/1000000:.1f}M\n")
                f.write(f"  Efficiency Score: {route_data['efficiency_score']:.2f}/10\n")
                f.write(f"  Peak Hour Demand: {route_data['demand']['peak_hour']:,} passengers\n")
                f.write(f"  Recommended Train: {route_data['train_recommendation']}\n")
                f.write(f"  Terrain Difficulty: {route_data['avg_difficulty']:.2f}\n")
                
                # ML recommendations
                recommendations = self.ml_system.recommend_improvements(route_data)
                if recommendations:
                    f.write("  Recommendations:\n")
                    for rec in recommendations:
                        f.write(f"    - {rec}\n")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Railway Network Planning AI System')
    parser.add_argument('--csv', required=True, help='CSV file with cities')
    parser.add_argument('--budget', type=float, default=1000000000, help='Budget in euros')
    parser.add_argument('--output', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize system
    planner = RailwayPlannerSystem()
    
    # Train ML models (in real implementation, load pre-trained models)
    logger.info("Training ML models...")
    planner.ml_system.train_models([])
    
    # Load cities
    cities = planner.load_cities_from_csv(args.csv)
    if not cities:
        logger.error("Failed to load cities from CSV")
        return
    
    # Plan network
    network_plan = planner.plan_optimal_network(args.budget)
    
    # Generate reports
    reports = planner.generate_reports(network_plan, args.output)
    
    logger.info("Railway network planning completed successfully!")
    logger.info(f"Check {args.output}/ directory for results")

if __name__ == "__main__":
    main()