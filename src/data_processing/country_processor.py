import pandas as pd
import geopandas as gpd
import requests
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import time
from dataclasses import dataclass
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

@dataclass
class CountryMetrics:
    """Metrics for a country's railway system"""
    total_network_length: float
    station_count: int
    electrification_rate: float
    high_speed_coverage: float
    population_density: float
    terrain_difficulty: float
    economic_index: float

class CountryProcessor:
    """Process and enrich country-specific railway data"""
    
    def __init__(self, config_path: str = "config/countries_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        self.geocoder = Nominatim(user_agent="railway_optimizer")
        
        # Country-specific railway characteristics
        self.railway_characteristics = {
            'germany': {
                'gauge': 1435,  # Standard gauge in mm
                'voltage': [15000, 25000],  # AC voltage
                'max_speed': 320,  # km/h
                'network_density': 'high',
                'terrain_adaptation': 'excellent',
                'electrification_standard': 'high'
            },
            'switzerland': {
                'gauge': 1435,
                'voltage': [15000],
                'max_speed': 200,
                'network_density': 'very_high',
                'terrain_adaptation': 'world_class',
                'electrification_standard': 'very_high'
            },
            'japan': {
                'gauge': [1435, 1067],  # Standard and Cape gauge
                'voltage': [1500, 25000],  # DC and AC
                'max_speed': 320,
                'network_density': 'high',
                'terrain_adaptation': 'excellent',
                'electrification_standard': 'high'
            },
            'france': {
                'gauge': 1435,
                'voltage': [1500, 25000],
                'max_speed': 320,
                'network_density': 'medium',
                'terrain_adaptation': 'good',
                'electrification_standard': 'high'
            },
            'netherlands': {
                'gauge': 1435,
                'voltage': [1500],
                'max_speed': 160,
                'network_density': 'very_high',
                'terrain_adaptation': 'excellent',  # Flat terrain
                'electrification_standard': 'complete'
            },
            'austria': {
                'gauge': 1435,
                'voltage': [15000],
                'max_speed': 230,
                'network_density': 'medium',
                'terrain_adaptation': 'excellent',
                'electrification_standard': 'high'
            },
            'sweden': {
                'gauge': 1435,
                'voltage': [15000],
                'max_speed': 200,
                'network_density': 'low',
                'terrain_adaptation': 'good',
                'electrification_standard': 'medium'
            },
            'denmark': {
                'gauge': 1435,
                'voltage': [25000],
                'max_speed': 180,
                'network_density': 'medium',
                'terrain_adaptation': 'excellent',  # Flat terrain
                'electrification_standard': 'high'
            }
        }
    
    def get_country_bounds(self, country: str) -> Tuple[float, float, float, float]:
        """Get country bounding box coordinates"""
        
        if country not in self.config['countries']:
            raise ValueError(f"Country {country} not in configuration")
        
        bbox_str = self.config['countries'][country]['bbox']
        bbox = [float(x) for x in bbox_str.split(',')]
        
        return tuple(bbox)  # south, west, north, east
    
    def get_country_info(self, country: str) -> Dict:
        """Get comprehensive country information"""
        
        if country not in self.config['countries']:
            raise ValueError(f"Country {country} not in configuration")
        
        country_config = self.config['countries'][country]
        
        # Basic info from config
        info = {
            'name': country_config['name'],
            'bbox': self.get_country_bounds(country),
            'railway_operators': country_config['railway_operators'],
            'train_types': country_config['train_types'],
            'terrain_types': country_config['terrain_types']
        }
        
        # Add railway characteristics
        if country in self.railway_characteristics:
            info['railway_characteristics'] = self.railway_characteristics[country]
        
        # Add population and area (approximate)
        country_stats = self._get_country_statistics(country)
        info.update(country_stats)
        
        return info
    
    def _get_country_statistics(self, country: str) -> Dict:
        """Get basic country statistics"""
        
        # Approximate statistics (in real implementation, fetch from APIs)
        stats = {
            'germany': {
                'population': 83240000,
                'area_km2': 357022,
                'gdp_per_capita': 46259,
                'rail_passengers_per_year': 2800000000,
                'rail_network_length_km': 33400
            },
            'switzerland': {
                'population': 8703000,
                'area_km2': 41285,
                'gdp_per_capita': 81867,
                'rail_passengers_per_year': 500000000,
                'rail_network_length_km': 5200
            },
            'japan': {
                'population': 125800000,
                'area_km2': 377975,
                'gdp_per_capita': 39285,
                'rail_passengers_per_year': 40000000000,
                'rail_network_length_km': 20000
            },
            'france': {
                'population': 67750000,
                'area_km2': 643801,
                'gdp_per_capita': 38625,
                'rail_passengers_per_year': 5000000000,
                'rail_network_length_km': 28000
            },
            'netherlands': {
                'population': 17530000,
                'area_km2': 41543,
                'gdp_per_capita': 52331,
                'rail_passengers_per_year': 1300000000,
                'rail_network_length_km': 3200
            },
            'austria': {
                'population': 8955000,
                'area_km2': 83879,
                'gdp_per_capita': 45436,
                'rail_passengers_per_year': 280000000,
                'rail_network_length_km': 5800
            },
            'sweden': {
                'population': 10350000,
                'area_km2': 450295,
                'gdp_per_capita': 51925,
                'rail_passengers_per_year': 200000000,
                'rail_network_length_km': 11900
            },
            'denmark': {
                'population': 5850000,
                'area_km2': 43094,
                'gdp_per_capita': 60170,
                'rail_passengers_per_year': 190000000,
                'rail_network_length_km': 2600
            }
        }
        
        return stats.get(country, {})
    
    def calculate_country_metrics(self, country: str, 
                                 network_data: gpd.GeoDataFrame,
                                 station_data: gpd.GeoDataFrame) -> CountryMetrics:
        """Calculate comprehensive metrics for a country's railway system"""
        
        country_info = self.get_country_info(country)
        
        # Network metrics
        if not network_data.empty:
            total_length = network_data.geometry.length.sum() * 111  # Convert to km
            electrified_segments = network_data[network_data.get('electrified', '') == 'yes']
            electrification_rate = len(electrified_segments) / len(network_data) if len(network_data) > 0 else 0
            
            # High speed coverage
            high_speed_segments = network_data[
                network_data.get('railway_type', '').isin(['high_speed']) |
                network_data.get('maxspeed', '').str.contains('200|250|300|320', na=False)
            ]
            high_speed_coverage = len(high_speed_segments) / len(network_data) if len(network_data) > 0 else 0
        else:
            total_length = 0
            electrification_rate = 0
            high_speed_coverage = 0
        
        # Station count
        station_count = len(station_data) if not station_data.empty else 0
        
        # Population density
        population = country_info.get('population', 0)
        area = country_info.get('area_km2', 1)
        population_density = population / area if area > 0 else 0
        
        # Terrain difficulty (based on terrain types)
        terrain_difficulty = self._calculate_terrain_difficulty(country_info.get('terrain_types', []))
        
        # Economic index (normalized GDP per capita)
        gdp_per_capita = country_info.get('gdp_per_capita', 0)
        economic_index = min(gdp_per_capita / 50000, 1.0)  # Normalize to 0-1
        
        return CountryMetrics(
            total_network_length=total_length,
            station_count=station_count,
            electrification_rate=electrification_rate,
            high_speed_coverage=high_speed_coverage,
            population_density=population_density,
            terrain_difficulty=terrain_difficulty,
            economic_index=economic_index
        )
    
    def _calculate_terrain_difficulty(self, terrain_types: List[str]) -> float:
        """Calculate terrain difficulty score (0-1, higher = more difficult)"""
        
        difficulty_scores = {
            'flat': 0.1,
            'coastal': 0.2,
            'hilly': 0.5,
            'mountainous': 0.9,
            'forest': 0.3,
            'urban': 0.4
        }
        
        if not terrain_types:
            return 0.5  # Default moderate difficulty
        
        scores = [difficulty_scores.get(terrain, 0.5) for terrain in terrain_types]
        return sum(scores) / len(scores)
    
    def enrich_network_data(self, network_data: gpd.GeoDataFrame, country: str) -> gpd.GeoDataFrame:
        """Enrich network data with country-specific information"""
        
        if network_data.empty:
            return network_data
        
        enriched_data = network_data.copy()
        
        # Add country information
        enriched_data['country'] = country
        
        # Add railway characteristics
        if country in self.railway_characteristics:
            chars = self.railway_characteristics[country]
            enriched_data['standard_gauge'] = chars['gauge']
            enriched_data['standard_voltage'] = str(chars['voltage'])
            enriched_data['max_country_speed'] = chars['max_speed']
            enriched_data['network_density'] = chars['network_density']
            enriched_data['terrain_adaptation'] = chars['terrain_adaptation']
        
        # Classify routes by importance
        enriched_data['route_importance'] = enriched_data.apply(
            lambda row: self._classify_route_importance(row, country), axis=1
        )
        
        # Add cost estimation
        enriched_data['estimated_cost_per_km'] = enriched_data.apply(
            lambda row: self._estimate_construction_cost(row, country), axis=1
        )
        
        return enriched_data
    
    def _classify_route_importance(self, route_row: pd.Series, country: str) -> str:
        """Classify route importance based on characteristics"""
        
        # Get route characteristics
        railway_type = route_row.get('railway_type', 'rail')
        usage = route_row.get('usage', 'main')
        service = route_row.get('service', 'passenger')
        maxspeed = route_row.get('maxspeed', '')
        
        # High importance: High-speed, main lines
        if railway_type == 'high_speed' or 'high_speed' in str(service):
            return 'critical'
        
        # Parse speed for classification
        speed = 0
        if maxspeed:
            try:
                speed = int(str(maxspeed).replace('km/h', '').replace('mph', ''))
            except:
                pass
        
        if speed >= 200:
            return 'critical'
        elif speed >= 120 or usage == 'main':
            return 'high'
        elif railway_type in ['rail', 'light_rail'] and usage == 'branch':
            return 'medium'
        else:
            return 'low'
    
    def _estimate_construction_cost(self, route_row: pd.Series, country: str) -> float:
        """Estimate construction cost per km based on route characteristics"""
        
        # Base cost per km in millions USD
        base_costs = {
            'germany': 15.0,
            'switzerland': 25.0,  # Higher due to terrain
            'japan': 20.0,
            'france': 12.0,
            'netherlands': 8.0,   # Lower due to flat terrain
            'austria': 18.0,
            'sweden': 10.0,
            'denmark': 9.0
        }
        
        base_cost = base_costs.get(country, 12.0)
        
        # Adjust based on railway type
        railway_type = route_row.get('railway_type', 'rail')
        if railway_type == 'high_speed':
            base_cost *= 2.0
        elif railway_type == 'subway':
            base_cost *= 3.0
        elif railway_type == 'light_rail':
            base_cost *= 0.7
        
        # Adjust based on electrification
        if route_row.get('electrified', 'no') == 'yes':
            base_cost *= 1.3
        
        return base_cost
    
    def compare_countries(self, countries: List[str], 
                         network_data: Dict[str, gpd.GeoDataFrame],
                         station_data: Dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
        """Compare multiple countries' railway systems"""
        
        comparison_data = []
        
        for country in countries:
            if country not in network_data or country not in station_data:
                self.logger.warning(f"Missing data for {country}")
                continue
            
            try:
                metrics = self.calculate_country_metrics(
                    country, 
                    network_data[country], 
                    station_data[country]
                )
                
                country_info = self.get_country_info(country)
                
                comparison_data.append({
                    'country': country,
                    'country_name': country_info['name'],
                    'population': country_info.get('population', 0),
                    'area_km2': country_info.get('area_km2', 0),
                    'network_length_km': metrics.total_network_length,
                    'station_count': metrics.station_count,
                    'electrification_rate': metrics.electrification_rate,
                    'high_speed_coverage': metrics.high_speed_coverage,
                    'population_density': metrics.population_density,
                    'terrain_difficulty': metrics.terrain_difficulty,
                    'economic_index': metrics.economic_index,
                    'network_density_per_1000km2': metrics.total_network_length / (country_info.get('area_km2', 1) / 1000),
                    'stations_per_1000km_network': metrics.station_count / (metrics.total_network_length / 1000) if metrics.total_network_length > 0 else 0
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {country}: {e}")
                continue
        
        return pd.DataFrame(comparison_data)
    
    def save_country_analysis(self, countries: List[str], 
                             output_dir: str = "data/processed/countries"):
        """Save comprehensive country analysis"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load network and station data
        network_data = {}
        station_data = {}
        
        for country in countries:
            try:
                # Load network data
                network_file = Path(f"data/input/railways/existing_networks/{country}/{country}_network.geojson")
                if network_file.exists():
                    network_data[country] = gpd.read_file(network_file)
                else:
                    network_data[country] = gpd.GeoDataFrame()
                
                # Load station data
                station_file = Path(f"data/input/railways/existing_networks/{country}/{country}_stations.geojson")
                if station_file.exists():
                    station_data[country] = gpd.read_file(station_file)
                else:
                    station_data[country] = gpd.GeoDataFrame()
                
            except Exception as e:
                self.logger.error(f"Error loading data for {country}: {e}")
                network_data[country] = gpd.GeoDataFrame()
                station_data[country] = gpd.GeoDataFrame()
        
        # Create comparison
        comparison = self.compare_countries(countries, network_data, station_data)
        
        if not comparison.empty:
            # Save comparison
            comparison.to_csv(output_path / "country_comparison.csv", index=False)
            
            # Save detailed analysis for each country
            for country in countries:
                if country in network_data:
                    enriched_network = self.enrich_network_data(network_data[country], country)
                    enriched_network.to_file(output_path / f"{country}_enriched_network.geojson", driver='GeoJSON')
        
        self.logger.info(f"Country analysis saved to {output_path}")
        return comparison

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create processor
    processor = CountryProcessor()
    
    # Test with sample countries
    countries = ['germany', 'switzerland', 'japan']
    
    # Save analysis
    comparison = processor.save_country_analysis(countries)
    
    if not comparison.empty:
        print("\n=== Country Railway System Comparison ===")
        print(comparison.to_string(index=False))
    else:
        print("No comparison data generated")