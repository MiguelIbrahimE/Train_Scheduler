import requests
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import time
import logging
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path

class RailwayNetworkExtractor:
    """Extract railway network data from OpenStreetMap using Overpass API"""
    
    def __init__(self, config_path: str = "config/countries_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.overpass_url = self.config['apis']['overpass']['base_url']
        self.timeout = self.config['apis']['overpass']['timeout']
        self.logger = logging.getLogger(__name__)
        
    def extract_country_network(self, country: str) -> gpd.GeoDataFrame:
        """Extract railway network for a specific country"""
        
        if country not in self.config['countries']:
            raise ValueError(f"Country {country} not in config")
            
        country_config = self.config['countries'][country]
        bbox = country_config['bbox']
        
        # Build Overpass query for railway lines
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["railway"~"rail|light_rail|subway|tram|narrow_gauge"]["railway"!="abandoned"]["railway"!="disused"]["railway"!="construction"]({bbox});
          relation["railway"~"rail|light_rail|subway|tram|narrow_gauge"]["railway"!="abandoned"]["railway"!="disused"]["railway"!="construction"]({bbox});
        );
        out geom;
        """
        
        self.logger.info(f"Fetching railway network for {country}...")
        
        try:
            response = requests.get(self.overpass_url, params={'data': query}, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            # Process ways into LineStrings
            lines = []
            for element in data['elements']:
                if element['type'] == 'way' and 'geometry' in element:
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    if len(coords) > 1:
                        tags = element.get('tags', {})
                        lines.append({
                            'geometry': LineString(coords),
                            'osm_id': element['id'],
                            'railway_type': tags.get('railway', 'rail'),
                            'usage': tags.get('usage', 'main'),
                            'service': tags.get('service', 'passenger'),
                            'electrified': tags.get('electrified', 'no'),
                            'maxspeed': tags.get('maxspeed', None),
                            'gauge': tags.get('gauge', None),
                            'operator': tags.get('operator', None),
                            'name': tags.get('name', None),
                            'ref': tags.get('ref', None),
                            'frequency': tags.get('frequency', None),
                            'voltage': tags.get('voltage', None),
                            'country': country
                        })
            
            gdf = gpd.GeoDataFrame(lines, crs='EPSG:4326')
            self.logger.info(f"Extracted {len(gdf)} railway segments for {country}")
            return gdf
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data for {country}: {e}")
            return gpd.GeoDataFrame()
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON for {country}: {e}")
            return gpd.GeoDataFrame()
    
    def extract_country_stations(self, country: str) -> gpd.GeoDataFrame:
        """Extract railway stations for a specific country"""
        
        if country not in self.config['countries']:
            raise ValueError(f"Country {country} not in config")
            
        country_config = self.config['countries'][country]
        bbox = country_config['bbox']
        
        # Build Overpass query for stations
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          node["railway"="station"]({bbox});
          node["railway"="halt"]({bbox});
          node["public_transport"="station"]["railway"]({bbox});
        );
        out;
        """
        
        self.logger.info(f"Fetching railway stations for {country}...")
        
        try:
            response = requests.get(self.overpass_url, params={'data': query}, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            # Process nodes into Points
            stations = []
            for element in data['elements']:
                if element['type'] == 'node':
                    tags = element.get('tags', {})
                    stations.append({
                        'geometry': Point(element['lon'], element['lat']),
                        'osm_id': element['id'],
                        'name': tags.get('name', 'Unknown'),
                        'railway': tags.get('railway', 'station'),
                        'operator': tags.get('operator', None),
                        'uic_ref': tags.get('uic_ref', None),
                        'iata': tags.get('iata', None),
                        'wheelchair': tags.get('wheelchair', None),
                        'public_transport': tags.get('public_transport', None),
                        'network': tags.get('network', None),
                        'platforms': tags.get('platforms', None),
                        'country': country
                    })
            
            gdf = gpd.GeoDataFrame(stations, crs='EPSG:4326')
            self.logger.info(f"Extracted {len(gdf)} railway stations for {country}")
            return gdf
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching stations for {country}: {e}")
            return gpd.GeoDataFrame()
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON for stations {country}: {e}")
            return gpd.GeoDataFrame()
    
    def classify_railway_type(self, row: pd.Series) -> str:
        """Classify railway into high_speed, intercity, regional, or commuter"""
        
        railway_type = row.get('railway_type', 'rail')
        usage = row.get('usage', 'main')
        service = row.get('service', 'passenger')
        maxspeed = row.get('maxspeed', None)
        
        # Speed-based classification
        if maxspeed:
            try:
                speed = int(maxspeed.replace('km/h', '').replace('mph', ''))
                if speed >= 250:
                    return 'high_speed'
                elif speed >= 160:
                    return 'intercity'
                elif speed >= 100:
                    return 'regional'
                else:
                    return 'commuter'
            except:
                pass
        
        # Usage-based classification
        if usage in ['main', 'branch']:
            if service == 'high_speed':
                return 'high_speed'
            elif service in ['long_distance', 'passenger']:
                return 'intercity'
            else:
                return 'regional'
        elif usage == 'branch':
            return 'regional'
        else:
            return 'commuter'
    
    def save_network_data(self, country: str, output_dir: str = "data/input/railways/existing_networks"):
        """Save railway network and station data for a country"""
        
        # Create output directory
        country_dir = Path(output_dir) / country
        country_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract and save network
        network = self.extract_country_network(country)
        if not network.empty:
            # Add classification
            network['classification'] = network.apply(self.classify_railway_type, axis=1)
            
            # Save main network
            network.to_file(country_dir / f"{country}_network.geojson", driver='GeoJSON')
            
            # Save by train type
            for train_type in ['high_speed', 'intercity', 'regional', 'commuter']:
                subset = network[network['classification'] == train_type]
                if not subset.empty:
                    subset.to_file(country_dir / f"{train_type}_routes.geojson", driver='GeoJSON')
        
        # Extract and save stations
        stations = self.extract_country_stations(country)
        if not stations.empty:
            stations.to_file(country_dir / f"{country}_stations.geojson", driver='GeoJSON')
            
            # Save as CSV for easier processing
            stations_df = stations.drop('geometry', axis=1)
            stations_df['lat'] = stations.geometry.y
            stations_df['lon'] = stations.geometry.x
            stations_df.to_csv(country_dir.parent.parent / "station_data" / country / f"{country}_stations.csv", index=False)
        
        self.logger.info(f"Saved railway data for {country} to {country_dir}")
        
        return network, stations
    
    def extract_all_countries(self, output_dir: str = "data/input/railways/existing_networks"):
        """Extract railway data for all configured countries"""
        
        results = {}
        
        for country in self.config['countries'].keys():
            try:
                self.logger.info(f"Processing {country}...")
                network, stations = self.save_network_data(country, output_dir)
                results[country] = {
                    'network_count': len(network),
                    'station_count': len(stations),
                    'success': True
                }
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Failed to process {country}: {e}")
                results[country] = {
                    'network_count': 0,
                    'station_count': 0,
                    'success': False,
                    'error': str(e)
                }
        
        return results

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create extractor
    extractor = RailwayNetworkExtractor()
    
    # Extract data for all countries
    results = extractor.extract_all_countries()
    
    # Print summary
    print("\n=== Extraction Summary ===")
    for country, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {country}: {result['network_count']} network segments, {result['station_count']} stations")
        if not result['success']:
            print(f"   Error: {result.get('error', 'Unknown error')}")