import pandas as pd
import geopandas as gpd
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Optional, Union
import requests
import json

class DataLoader:
    """Load and validate data from various sources"""
    
    def __init__(self, config_path: str = "config/countries_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data")
        
    def load_city_data(self, country: str) -> pd.DataFrame:
        """Load city data for a specific country"""
        
        city_file = self.data_dir / "input" / "cities" / country / f"{country}_cities.csv"
        
        if not city_file.exists():
            self.logger.warning(f"City file not found for {country}, creating sample data")
            return self._create_sample_cities(country)
        
        try:
            cities_df = pd.read_csv(city_file)
            return self._validate_city_data(cities_df, country)
        except Exception as e:
            self.logger.error(f"Error loading city data for {country}: {e}")
            return pd.DataFrame()
    
    def _create_sample_cities(self, country: str) -> pd.DataFrame:
        """Create sample city data for a country using major cities"""
        
        # Major cities for each country
        major_cities = {
            'germany': [
                ('Berlin', 52.5200, 13.4050, 3669491),
                ('Hamburg', 53.5511, 9.9937, 1899160),
                ('Munich', 48.1351, 11.5820, 1471508),
                ('Cologne', 50.9375, 6.9603, 1085664),
                ('Frankfurt', 50.1109, 8.6821, 753056),
                ('Stuttgart', 48.7758, 9.1829, 630305),
                ('Düsseldorf', 51.2277, 6.7735, 619294),
                ('Dortmund', 51.5136, 7.4653, 588250),
                ('Essen', 51.4556, 7.0116, 579432),
                ('Leipzig', 51.3397, 12.3731, 593145)
            ],
            'switzerland': [
                ('Zurich', 47.3769, 8.5417, 415367),
                ('Geneva', 46.2044, 6.1432, 201818),
                ('Basel', 47.5596, 7.5886, 177595),
                ('Bern', 46.9481, 7.4474, 133883),
                ('Lausanne', 46.5197, 6.6323, 139111),
                ('Winterthur', 47.4979, 8.7240, 111851),
                ('Lucerne', 47.0502, 8.3093, 81592),
                ('St. Gallen', 47.4245, 9.3767, 75833),
                ('Lugano', 46.0037, 8.9511, 63668),
                ('Biel', 47.1368, 7.2446, 55203)
            ],
            'japan': [
                ('Tokyo', 35.6762, 139.6503, 37400068),
                ('Osaka', 34.6937, 135.5023, 19222665),
                ('Nagoya', 35.1815, 136.9066, 10110000),
                ('Sapporo', 43.0642, 141.3469, 2374534),
                ('Fukuoka', 33.5904, 130.4017, 2571000),
                ('Kobe', 34.6901, 135.1956, 1518870),
                ('Kyoto', 35.0116, 135.7681, 1475183),
                ('Yokohama', 35.4437, 139.6380, 3748781),
                ('Hiroshima', 34.3853, 132.4553, 1194034),
                ('Sendai', 38.2682, 140.8694, 1096704)
            ],
            'france': [
                ('Paris', 48.8566, 2.3522, 10958000),
                ('Lyon', 45.7640, 4.8357, 2280845),
                ('Marseille', 43.2965, 5.3698, 1879601),
                ('Toulouse', 43.6047, 1.4442, 1360829),
                ('Nice', 43.7102, 7.2620, 1006201),
                ('Nantes', 47.2184, -1.5536, 972828),
                ('Montpellier', 43.6110, 3.8767, 607896),
                ('Strasbourg', 48.5734, 7.7521, 505272),
                ('Bordeaux', 44.8378, -0.5792, 1363711),
                ('Lille', 50.6292, 3.0573, 1510079)
            ],
            'netherlands': [
                ('Amsterdam', 52.3676, 4.9041, 872757),
                ('Rotterdam', 51.9244, 4.4777, 651446),
                ('The Hague', 52.0705, 4.3007, 547757),
                ('Utrecht', 52.0907, 5.1214, 361966),
                ('Eindhoven', 51.4416, 5.4697, 234235),
                ('Groningen', 53.2194, 6.5665, 232922),
                ('Tilburg', 51.5656, 5.0913, 219800),
                ('Almere', 52.3508, 5.2647, 213090),
                ('Breda', 51.5719, 4.7683, 184403),
                ('Nijmegen', 51.8426, 5.8518, 179073)
            ],
            'austria': [
                ('Vienna', 48.2082, 16.3738, 1911191),
                ('Graz', 47.0707, 15.4395, 328276),
                ('Linz', 48.3069, 14.2858, 206604),
                ('Salzburg', 47.8095, 13.0550, 155021),
                ('Innsbruck', 47.2692, 11.4041, 132236),
                ('Klagenfurt', 46.6264, 14.3083, 100817),
                ('Villach', 46.6111, 13.8558, 62036),
                ('Wels', 48.1598, 14.0307, 62773),
                ('Sankt Pölten', 48.2058, 15.6232, 55439),
                ('Dornbirn', 47.4124, 9.7209, 49845)
            ],
            'sweden': [
                ('Stockholm', 59.3293, 18.0686, 1515017),
                ('Gothenburg', 57.7089, 11.9746, 599011),
                ('Malmö', 55.6059, 13.0007, 344166),
                ('Uppsala', 59.8586, 17.6389, 168096),
                ('Västerås', 59.6099, 16.5448, 127799),
                ('Örebro', 59.2753, 15.2134, 124207),
                ('Linköping', 58.4108, 15.6214, 115508),
                ('Helsingborg', 56.0465, 12.6945, 112496),
                ('Jönköping', 57.7826, 14.1618, 95071),
                ('Norrköping', 58.5877, 16.1924, 95618)
            ],
            'denmark': [
                ('Copenhagen', 55.6761, 12.5683, 1346485),
                ('Aarhus', 56.1629, 10.2039, 285273),
                ('Odense', 55.4038, 10.4024, 180760),
                ('Aalborg', 57.0488, 9.9217, 143035),
                ('Esbjerg', 55.4669, 8.4520, 71886),
                ('Randers', 56.4607, 10.0369, 62342),
                ('Kolding', 55.4904, 9.4721, 61121),
                ('Horsens', 55.8607, 9.8501, 59449),
                ('Vejle', 55.7094, 9.5357, 60213),
                ('Roskilde', 55.6418, 12.0879, 51916)
            ]
        }
        
        if country not in major_cities:
            self.logger.warning(f"No sample cities defined for {country}")
            return pd.DataFrame()
        
        cities_data = major_cities[country]
        cities_df = pd.DataFrame(cities_data, columns=['city', 'lat', 'lon', 'population'])
        
        # Save to file
        output_file = self.data_dir / "input" / "cities" / country / f"{country}_cities.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cities_df.to_csv(output_file, index=False)
        
        self.logger.info(f"Created sample cities file for {country}")
        return cities_df
    
    def _validate_city_data(self, cities_df: pd.DataFrame, country: str) -> pd.DataFrame:
        """Validate city data format and contents"""
        
        required_columns = ['city', 'lat', 'lon', 'population']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in cities_df.columns]
        if missing_columns:
            self.logger.error(f"Missing columns in {country} cities data: {missing_columns}")
            return pd.DataFrame()
        
        # Validate data types
        try:
            cities_df['lat'] = pd.to_numeric(cities_df['lat'])
            cities_df['lon'] = pd.to_numeric(cities_df['lon'])
            cities_df['population'] = pd.to_numeric(cities_df['population'])
        except ValueError as e:
            self.logger.error(f"Invalid data types in {country} cities data: {e}")
            return pd.DataFrame()
        
        # Validate coordinate ranges
        country_config = self.config['countries'][country]
        bbox = [float(x) for x in country_config['bbox'].split(',')]
        south, west, north, east = bbox
        
        # Filter cities within country bounds
        valid_cities = cities_df[
            (cities_df['lat'] >= south) & (cities_df['lat'] <= north) &
            (cities_df['lon'] >= west) & (cities_df['lon'] <= east)
        ]
        
        if len(valid_cities) < len(cities_df):
            self.logger.warning(f"Filtered {len(cities_df) - len(valid_cities)} cities outside {country} bounds")
        
        return valid_cities
    
    def load_railway_network(self, country: str) -> gpd.GeoDataFrame:
        """Load railway network data for a country"""
        
        network_file = self.data_dir / "input" / "railways" / "existing_networks" / country / f"{country}_network.geojson"
        
        if not network_file.exists():
            self.logger.warning(f"Railway network file not found for {country}")
            return gpd.GeoDataFrame()
        
        try:
            network_gdf = gpd.read_file(network_file)
            return network_gdf
        except Exception as e:
            self.logger.error(f"Error loading railway network for {country}: {e}")
            return gpd.GeoDataFrame()
    
    def load_railway_stations(self, country: str) -> gpd.GeoDataFrame:
        """Load railway stations data for a country"""
        
        stations_file = self.data_dir / "input" / "railways" / "existing_networks" / country / f"{country}_stations.geojson"
        
        if not stations_file.exists():
            self.logger.warning(f"Railway stations file not found for {country}")
            return gpd.GeoDataFrame()
        
        try:
            stations_gdf = gpd.read_file(stations_file)
            return stations_gdf
        except Exception as e:
            self.logger.error(f"Error loading railway stations for {country}: {e}")
            return gpd.GeoDataFrame()
    
    def load_economic_data(self) -> Dict[str, pd.DataFrame]:
        """Load economic data (population, GDP, etc.)"""
        
        economic_data = {}
        economic_dir = self.data_dir / "input" / "economic"
        
        for file_path in economic_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                economic_data[file_path.stem] = df
            except Exception as e:
                self.logger.error(f"Error loading economic data from {file_path}: {e}")
        
        return economic_data
    
    def enrich_cities_with_online_data(self, cities_df: pd.DataFrame, country: str) -> pd.DataFrame:
        """Enrich city data with online sources (GDP, tourism, etc.)"""
        
        # Add country column
        cities_df['country'] = country
        
        # Add estimated GDP per capita based on country averages
        country_gdp = {
            'germany': 46259,
            'switzerland': 81867,
            'japan': 39285,
            'france': 38625,
            'netherlands': 52331,
            'austria': 45436,
            'sweden': 51925,
            'denmark': 60170
        }
        
        cities_df['gdp_per_capita'] = country_gdp.get(country, 30000)
        
        # Add tourism index (mock data - in real implementation, fetch from APIs)
        cities_df['tourism_index'] = cities_df['population'] / 100000  # Simple proxy
        
        # Add commuter index based on population
        cities_df['commuter_index'] = (cities_df['population'] / cities_df['population'].max()) * 100
        
        return cities_df
    
    def load_all_country_data(self, country: str) -> Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]:
        """Load all data for a specific country"""
        
        data = {}
        
        # Load cities
        cities = self.load_city_data(country)
        if not cities.empty:
            cities = self.enrich_cities_with_online_data(cities, country)
            data['cities'] = cities
        
        # Load railway network
        network = self.load_railway_network(country)
        if not network.empty:
            data['railway_network'] = network
        
        # Load railway stations
        stations = self.load_railway_stations(country)
        if not stations.empty:
            data['railway_stations'] = stations
        
        return data
    
    def load_all_countries_data(self) -> Dict[str, Dict]:
        """Load data for all configured countries"""
        
        all_data = {}
        
        for country in self.config['countries'].keys():
            self.logger.info(f"Loading data for {country}...")
            country_data = self.load_all_country_data(country)
            
            if country_data:
                all_data[country] = country_data
                self.logger.info(f"✅ Loaded {len(country_data)} datasets for {country}")
            else:
                self.logger.warning(f"❌ No data loaded for {country}")
        
        return all_data
    
    def save_processed_data(self, data: Dict, output_path: str):
        """Save processed data to files"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for country, country_data in data.items():
            country_dir = output_dir / country
            country_dir.mkdir(exist_ok=True)
            
            for dataset_name, dataset in country_data.items():
                if isinstance(dataset, gpd.GeoDataFrame):
                    dataset.to_file(country_dir / f"{dataset_name}.geojson", driver='GeoJSON')
                elif isinstance(dataset, pd.DataFrame):
                    dataset.to_csv(country_dir / f"{dataset_name}.csv", index=False)
        
        self.logger.info(f"Saved processed data to {output_dir}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data loader
    loader = DataLoader()
    
    # Load all data
    all_data = loader.load_all_countries_data()
    
    # Save processed data
    loader.save_processed_data(all_data, "data/processed/countries")
    
    # Print summary
    print("\n=== Data Loading Summary ===")
    for country, data in all_data.items():
        print(f"🏴 {country}:")
        for dataset_name, dataset in data.items():
            print(f"  - {dataset_name}: {len(dataset)} records")