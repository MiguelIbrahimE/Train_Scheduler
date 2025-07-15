import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import time
import logging
from typing import List, Tuple, Dict, Optional
import yaml
from pathlib import Path
import json

class TerrainAnalyzer:
    """Analyze terrain characteristics using OpenElevation API and OSM data"""
    
    def __init__(self, config_path: str = "config/countries_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.elevation_api = self.config['apis']['elevation']['base_url']
        self.rate_limit = self.config['apis']['elevation']['rate_limit']
        self.terrain_costs = self.config['terrain_costs']
        self.logger = logging.getLogger(__name__)
        
    def get_elevation_point(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation for a single point using OpenElevation API"""
        
        try:
            url = f"{self.elevation_api}/lookup"
            params = {'locations': f"{lat},{lon}"}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                return data['results'][0]['elevation']
            
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Error fetching elevation for {lat},{lon}: {e}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            self.logger.warning(f"Error parsing elevation data for {lat},{lon}: {e}")
        
        return None
    
    def get_elevation_batch(self, coordinates: List[Tuple[float, float]], batch_size: int = 50) -> List[Optional[float]]:
        """Get elevation for multiple points in batches"""
        
        elevations = []
        
        for i in range(0, len(coordinates), batch_size):
            batch = coordinates[i:i + batch_size]
            
            # Format coordinates for API
            locations = "|".join([f"{lat},{lon}" for lat, lon in batch])
            
            try:
                url = f"{self.elevation_api}/lookup"
                params = {'locations': locations}
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if 'results' in data:
                    batch_elevations = [result['elevation'] for result in data['results']]
                    elevations.extend(batch_elevations)
                else:
                    elevations.extend([None] * len(batch))
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error fetching elevation batch: {e}")
                elevations.extend([None] * len(batch))
        
        return elevations
    
    def get_route_elevation_profile(self, route_coords: List[Tuple[float, float]], 
                                   sample_distance: float = 1000) -> pd.DataFrame:
        """Get elevation profile for a route with regular sampling"""
        
        # Sample points along the route
        if len(route_coords) < 2:
            return pd.DataFrame()
        
        # Create LineString from coordinates
        line = LineString([(lon, lat) for lat, lon in route_coords])
        
        # Sample points every sample_distance meters
        distances = np.arange(0, line.length, sample_distance / 111000)  # rough conversion to degrees
        
        sampled_points = []
        for distance in distances:
            point = line.interpolate(distance)
            sampled_points.append((point.y, point.x))  # lat, lon
        
        # Add end point
        if sampled_points[-1] != route_coords[-1]:
            sampled_points.append(route_coords[-1])
        
        # Get elevations
        elevations = self.get_elevation_batch(sampled_points)
        
        # Create DataFrame
        profile_df = pd.DataFrame({
            'lat': [p[0] for p in sampled_points],
            'lon': [p[1] for p in sampled_points],
            'elevation': elevations,
            'distance_km': np.arange(len(sampled_points)) * sample_distance / 1000
        })
        
        # Fill missing elevations with interpolation
        profile_df['elevation'] = profile_df['elevation'].interpolate()
        
        return profile_df
    
    def calculate_terrain_metrics(self, elevation_profile: pd.DataFrame) -> Dict[str, float]:
        """Calculate terrain metrics from elevation profile"""
        
        if elevation_profile.empty or 'elevation' not in elevation_profile.columns:
            return {}
        
        elevations = elevation_profile['elevation'].dropna()
        
        if len(elevations) < 2:
            return {}
        
        # Basic metrics
        min_elevation = elevations.min()
        max_elevation = elevations.max()
        elevation_range = max_elevation - min_elevation
        mean_elevation = elevations.mean()
        
        # Calculate gradients
        distances = elevation_profile['distance_km'].values
        gradients = np.gradient(elevations, distances)
        
        # Gradient metrics
        mean_gradient = np.abs(gradients).mean()
        max_gradient = np.abs(gradients).max()
        steep_sections = np.sum(np.abs(gradients) > 0.03)  # >3% gradient
        
        # Terrain classification
        terrain_type = self.classify_terrain(elevation_range, mean_gradient, max_gradient)
        
        return {
            'min_elevation': min_elevation,
            'max_elevation': max_elevation,
            'elevation_range': elevation_range,
            'mean_elevation': mean_elevation,
            'mean_gradient': mean_gradient,
            'max_gradient': max_gradient,
            'steep_sections': steep_sections,
            'terrain_type': terrain_type,
            'terrain_cost_multiplier': self.terrain_costs.get(terrain_type, 1.0)
        }
    
    def classify_terrain(self, elevation_range: float, mean_gradient: float, max_gradient: float) -> str:
        """Classify terrain type based on elevation and gradient metrics"""
        
        if elevation_range > 1000 or max_gradient > 0.08:
            return 'mountainous'
        elif elevation_range > 300 or mean_gradient > 0.03:
            return 'hilly'
        elif elevation_range < 50 and mean_gradient < 0.01:
            return 'flat'
        else:
            return 'hilly'
    
    def calculate_construction_cost(self, terrain_metrics: Dict[str, float], 
                                  route_length_km: float) -> Dict[str, float]:
        """Calculate construction cost based on terrain characteristics"""
        
        # Base cost per km (in millions)
        base_cost_per_km = 10.0
        
        # Terrain multiplier
        terrain_multiplier = terrain_metrics.get('terrain_cost_multiplier', 1.0)
        
        # Gradient penalties
        gradient_penalty = 1.0
        if terrain_metrics.get('max_gradient', 0) > 0.05:  # >5% gradient
            gradient_penalty = 1.5
        if terrain_metrics.get('max_gradient', 0) > 0.08:  # >8% gradient
            gradient_penalty = 2.0
        
        # Elevation penalties (tunnels/bridges needed)
        elevation_penalty = 1.0
        if terrain_metrics.get('elevation_range', 0) > 500:
            elevation_penalty = 1.3
        if terrain_metrics.get('elevation_range', 0) > 1000:
            elevation_penalty = 1.8
        
        # Calculate total cost
        total_multiplier = terrain_multiplier * gradient_penalty * elevation_penalty
        total_cost = base_cost_per_km * route_length_km * total_multiplier
        
        return {
            'base_cost_millions': base_cost_per_km * route_length_km,
            'terrain_multiplier': terrain_multiplier,
            'gradient_penalty': gradient_penalty,
            'elevation_penalty': elevation_penalty,
            'total_multiplier': total_multiplier,
            'total_cost_millions': total_cost
        }
    
    def analyze_existing_route(self, route_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Analyze terrain characteristics of existing railway routes"""
        
        results = []
        
        for idx, row in route_gdf.iterrows():
            try:
                # Extract coordinates from LineString
                coords = list(row.geometry.coords)
                route_coords = [(lat, lon) for lon, lat in coords]
                
                # Get elevation profile
                elevation_profile = self.get_route_elevation_profile(route_coords)
                
                # Calculate terrain metrics
                terrain_metrics = self.calculate_terrain_metrics(elevation_profile)
                
                # Calculate route length
                route_length_km = row.geometry.length * 111  # rough conversion to km
                
                # Calculate construction cost
                cost_metrics = self.calculate_construction_cost(terrain_metrics, route_length_km)
                
                # Combine all metrics
                route_analysis = {
                    'route_id': idx,
                    'osm_id': row.get('osm_id', None),
                    'route_length_km': route_length_km,
                    **terrain_metrics,
                    **cost_metrics
                }
                
                results.append(route_analysis)
                
            except Exception as e:
                self.logger.error(f"Error analyzing route {idx}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def fetch_nimby_factors(self, bbox: str) -> gpd.GeoDataFrame:
        """Fetch NIMBY factors (protected areas, residential zones) from OSM"""
        
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Query for protected areas, residential zones, etc.
        query = f"""
        [out:json][timeout:60];
        (
          way["landuse"="residential"]({bbox});
          way["leisure"="nature_reserve"]({bbox});
          way["boundary"="protected_area"]({bbox});
          way["landuse"="forest"]({bbox});
          relation["boundary"="protected_area"]({bbox});
        );
        out geom;
        """
        
        try:
            response = requests.get(overpass_url, params={'data': query}, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            # Process areas
            areas = []
            for element in data['elements']:
                if element['type'] == 'way' and 'geometry' in element:
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    if len(coords) > 2:  # Need at least 3 points for polygon
                        tags = element.get('tags', {})
                        areas.append({
                            'geometry': LineString(coords),  # Convert to polygon if needed
                            'landuse': tags.get('landuse', None),
                            'leisure': tags.get('leisure', None),
                            'boundary': tags.get('boundary', None),
                            'protection_level': tags.get('protection_title', None)
                        })
            
            return gpd.GeoDataFrame(areas, crs='EPSG:4326')
            
        except Exception as e:
            self.logger.error(f"Error fetching NIMBY factors: {e}")
            return gpd.GeoDataFrame()
    
    def save_terrain_analysis(self, country: str, output_dir: str = "data/processed/terrain_cost_maps"):
        """Save terrain analysis for a country"""
        
        # Load existing network data
        network_file = Path(f"data/input/railways/existing_networks/{country}/{country}_network.geojson")
        
        if not network_file.exists():
            self.logger.error(f"Network file not found for {country}")
            return
        
        # Load network
        network_gdf = gpd.read_file(network_file)
        
        # Analyze terrain
        terrain_analysis = self.analyze_existing_route(network_gdf)
        
        # Save results
        output_path = Path(output_dir) / f"{country}_terrain_analysis.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        terrain_analysis.to_csv(output_path, index=False)
        
        self.logger.info(f"Saved terrain analysis for {country} to {output_path}")
        
        return terrain_analysis

if __name__ == "__main__":
    # Set up logging
    logging