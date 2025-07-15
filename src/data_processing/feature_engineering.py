import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from scipy.spatial.distance import cdist
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import yaml

class FeatureEngineer:
    """Engineer features for railway route optimization machine learning models"""
    
    def __init__(self, config_path: str = "config/countries_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
        # Feature importance weights
        self.feature_weights = {
            'population': 0.25,
            'distance': 0.20,
            'elevation': 0.15,
            'economic': 0.15,
            'connectivity': 0.10,
            'terrain': 0.10,
            'existing_rail': 0.05
        }
    
    def create_city_features(self, cities_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for cities"""
        
        features_df = cities_df.copy()
        
        # Basic demographic features
        features_df = self._add_population_features(features_df)
        
        # Economic features
        features_df = self._add_economic_features(features_df)
        
        # Geographic features
        features_df = self._add_geographic_features(features_df)
        
        # Connectivity features
        features_df = self._add_connectivity_features(features_df)
        
        # Clustering features
        features_df = self._add_clustering_features(features_df)
        
        return features_df
    
    def _add_population_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add population-based features"""
        
        df_pop = df.copy()
        
        if 'population' in df_pop.columns:
            # Log population (to handle skewness)
            df_pop['log_population'] = np.log1p(df_pop['population'])
            
            # Population rank
            df_pop['population_rank'] = df_pop['population'].rank(ascending=False)
            
            # Population percentile
            df_pop['population_percentile'] = df_pop['population'].rank(pct=True)
            
            # Population density proxy (population per unit area)
            if 'area' in df_pop.columns:
                df_pop['population_density'] = df_pop['population'] / (df_pop['area'] + 1)
            
            # Population categories
            df_pop['city_size_category'] = pd.cut(
                df_pop['population'],
                bins=[0, 50000, 200000, 500000, 1000000, float('inf')],
                labels=['small', 'medium', 'large', 'major', 'megacity']
            )
            
            # Population growth potential (based on current size)
            df_pop['growth_potential'] = df_pop['population'].apply(
                lambda x: 1.0 if x < 100000 else 0.8 if x < 500000 else 0.6 if x < 1000000 else 0.4
            )
        
        return df_pop
    
    def _add_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add economic features"""
        
        df_econ = df.copy()
        
        # GDP features
        if 'gdp_per_capita' in df_econ.columns:
            df_econ['log_gdp'] = np.log1p(df_econ['gdp_per_capita'])
            df_econ['gdp_rank'] = df_econ['gdp_per_capita'].rank(ascending=False)
            df_econ['gdp_percentile'] = df_econ['gdp_per_capita'].rank(pct=True)
        
        # Economic activity
        if 'economic_activity' in df_econ.columns:
            df_econ['economic_activity_norm'] = (
                df_econ['economic_activity'] - df_econ['economic_activity'].min()
            ) / (df_econ['economic_activity'].max() - df_econ['economic_activity'].min())
        
        # Tourism features
        if 'tourism_index' in df_econ.columns:
            df_econ['tourism_category'] = pd.cut(
                df_econ['tourism_index'],
                bins=[0, 3, 6, 8, 10],
                labels=['low', 'medium', 'high', 'very_high']
            )
            df_econ['tourism_weight'] = df_econ['tourism_index'] / 10.0
        
        # Combined economic score
        economic_cols = ['gdp_per_capita', 'economic_activity', 'tourism_index']
        available_cols = [col for col in economic_cols if col in df_econ.columns]
        
        if available_cols:
            # Normalize each component
            for col in available_cols:
                norm_col = f"{col}_norm"
                df_econ[norm_col] = (df_econ[col] - df_econ[col].min()) / (df_econ[col].max() - df_econ[col].min())
            
            # Calculate combined score
            norm_cols = [f"{col}_norm" for col in available_cols]
            df_econ['economic_score'] = df_econ[norm_cols].mean(axis=1)
        
        return df_econ
    
    def _add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geographic features"""
        
        df_geo = df.copy()
        
        if 'lat' in df_geo.columns and 'lon' in df_geo.columns:
            # Distance from center
            center_lat = df_geo['lat'].mean()
            center_lon = df_geo['lon'].mean()
            
            df_geo['distance_from_center'] = np.sqrt(
                (df_geo['lat'] - center_lat)**2 + (df_geo['lon'] - center_lon)**2
            )
            
            # Coastal proximity (rough approximation)
            df_geo['coastal_proximity'] = self._calculate_coastal_proximity(df_geo)
            
            # Regional classification
            df_geo['lat_band'] = pd.cut(
                df_geo['lat'],
                bins=5,
                labels=['south', 'south_mid', 'center', 'north_mid', 'north']
            )
            
            df_geo['lon_band'] = pd.cut(
                df_geo['lon'],
                bins=5,
                labels=['west', 'west_mid', 'center', 'east_mid', 'east']
            )
            
            # Elevation features (if available)
            if 'elevation' in df_geo.columns:
                df_geo['elevation_norm'] = (df_geo['elevation'] - df_geo['elevation'].min()) / (df_geo['elevation'].max() - df_geo['elevation'].min())
                df_geo['elevation_category'] = pd.cut(
                    df_geo['elevation'],
                    bins=[0, 200, 500, 1000, float('inf')],
                    labels=['lowland', 'hills', 'mountains', 'high_mountains']
                )
        
        return df_geo
    
    def _calculate_coastal_proximity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate coastal proximity score (0-1)"""
        
        # Simple approximation based on distance from country boundaries
        # In real implementation, this would use actual coastline data
        
        coastal_scores = []
        for _, row in df.iterrows():
            # Simplified coastal proximity based on lat/lon ranges
            lat, lon = row['lat'], row['lon']
            
            # Distance from boundaries (very rough approximation)
            boundary_distance = min(
                abs(lat - 90), abs(lat + 90),  # Polar distances
                abs(lon - 180), abs(lon + 180)  # Meridian distances
            )
            
            # Inverse relationship (closer to boundary = higher coastal score)
            coastal_score = max(0, 1 - (boundary_distance / 90))
            coastal_scores.append(coastal_score)
        
        return pd.Series(coastal_scores, index=df.index)
    
    def _add_connectivity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add connectivity features"""
        
        df_conn = df.copy()
        
        if 'lat' in df_conn.columns and 'lon' in df_conn.columns:
            # Calculate distance matrix
            coords = df_conn[['lat', 'lon']].values
            dist_matrix = cdist(coords, coords, metric='euclidean')
            
            # Nearest neighbor distances
            df_conn['nearest_neighbor_dist'] = np.array([
                np.min(dist_matrix[i][dist_matrix[i] > 0]) 
                for i in range(len(dist_matrix))
            ])
            
            # Average distance to top 5 nearest cities
            df_conn['avg_5_nearest_dist'] = np.array([
                np.mean(np.sort(dist_matrix[i])[1:6])  # Exclude self (index 0)
                for i in range(len(dist_matrix))
            ])
            
            # Connectivity score (inverse of average distance)
            df_conn['connectivity_score'] = 1 / (df_conn['avg_5_nearest_dist'] + 0.01)
            
            # Centrality measures
            df_conn['geographic_centrality'] = self._calculate_geographic_centrality(df_conn)
            
            # Cluster membership
            df_conn = self._add_spatial_clusters(df_conn)
        
        return df_conn
    
    def _calculate_geographic_centrality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate geographic centrality score"""
        
        # Create network graph based on distances
        coords = df[['lat', 'lon']].values
        dist_matrix = cdist(coords, coords, metric='euclidean')
        
        # Create graph where edges connect cities within threshold distance
        threshold = np.percentile(dist_matrix[dist_matrix > 0], 25)  # 25th percentile
        
        G = nx.Graph()
        for i in range(len(df)):
            G.add_node(i)
        
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                if dist_matrix[i][j] <= threshold:
                    G.add_edge(i, j, weight=1/dist_matrix[i][j])
        
        # Calculate centrality measures
        try:
            centrality = nx.closeness_centrality(G)
            return pd.Series([centrality.get(i, 0) for i in range(len(df))], index=df.index)
        except:
            return pd.Series(np.zeros(len(df)), index=df.index)
    
    def _add_spatial_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spatial clustering features"""
        
        df_cluster = df.copy()
        
        if 'lat' in df_cluster.columns and 'lon' in df_cluster.columns:
            # K-means clustering
            coords = df_cluster[['lat', 'lon']].values
            
            # Determine optimal number of clusters (max 8)
            n_clusters = min(8, max(2, len(df_cluster) // 5))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_cluster['spatial_cluster'] = kmeans.fit_predict(coords)
            
            # Cluster statistics
            cluster_stats = df_cluster.groupby('spatial_cluster').agg({
                'population': ['mean', 'sum', 'count'],
                'lat': ['mean'],
                'lon': ['mean']
            }).round(4)
            
            # Add cluster-level features
            df_cluster['cluster_population_mean'] = df_cluster['spatial_cluster'].map(
                cluster_stats[('population', 'mean')]
            )
            df_cluster['cluster_population_sum'] = df_cluster['spatial_cluster'].map(
                cluster_stats[('population', 'sum')]
            )
            df_cluster['cluster_size'] = df_cluster['spatial_cluster'].map(
                cluster_stats[('population', 'count')]
            )
            
            # Distance to cluster center
            cluster_centers = kmeans.cluster_centers_
            df_cluster['dist_to_cluster_center'] = [
                np.sqrt((row['lat'] - cluster_centers[row['spatial_cluster']][0])**2 +
                       (row['lon'] - cluster_centers[row['spatial_cluster']][1])**2)
                for _, row in df_cluster.iterrows()
            ]
        
        return df_cluster
    
    def _add_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add general clustering features"""
        
        df_clust = df.copy()
        
        # Select features for clustering
        cluster_features = []
        potential_features = ['population', 'gdp_per_capita', 'economic_activity', 'tourism_index']
        
        for feature in potential_features:
            if feature in df_clust.columns:
                cluster_features.append(feature)
        
        if len(cluster_features) >= 2:
            # Normalize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_clust[cluster_features])
            
            # K-means clustering
            n_clusters = min(5, max(2, len(df_clust) // 10))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_clust['economic_cluster'] = kmeans.fit_predict(scaled_features)
            
            # PCA for dimensionality reduction
            if len(cluster_features) > 2:
                pca = PCA(n_components=2)
                pca_features = pca.fit_transform(scaled_features)
                df_clust['pca_component_1'] = pca_features[:, 0]
                df_clust['pca_component_2'] = pca_features[:, 1]
                df_clust['pca_explained_variance'] = pca.explained_variance_ratio_.sum()
        
        return df_clust
    
    def create_route_features(self, city_pairs: List[Tuple[str, str]], 
                            cities_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for city pairs (potential routes)"""
        
        route_features = []
        
        for city1, city2 in city_pairs:
            # Get city data
            city1_data = cities_df[cities_df['city'] == city1]
            city2_data = cities_df[cities_df['city'] == city2]
            
            if city1_data.empty or city2_data.empty:
                continue
            
            city1_row = city1_data.iloc[0]
            city2_row = city2_data.iloc[0]
            
            # Basic route features
            features = {
                'city1': city1,
                'city2': city2,
                'route_id': f"{city1}_{city2}"
            }
            
            # Distance features
            features.update(self._calculate_distance_features(city1_row, city2_row))
            
            # Population features
            features.update(self._calculate_population_features(city1_row, city2_row))
            
            # Economic features
            features.update(self._calculate_economic_features(city1_row, city2_row))
            
            # Geographic features
            features.update(self._calculate_geographic_features(city1_row, city2_row))
            
            # Connectivity features
            features.update(self._calculate_connectivity_features(city1_row, city2_row))
            
            # Demand prediction features
            features.update(self._calculate_demand_features(city1_row, city2_row))
            
            route_features.append(features)
        
        return pd.DataFrame(route_features)
    
    def _calculate_distance_features(self, city1: pd.Series, city2: pd.Series) -> Dict:
        """Calculate distance-based features"""
        
        # Euclidean distance
        euclidean_dist = np.sqrt(
            (city1['lat'] - city2['lat'])**2 + (city1['lon'] - city2['lon'])**2
        )
        
        # Haversine distance (more accurate for geographic coordinates)
        haversine_dist = self._haversine_distance(
            city1['lat'], city1['lon'], city2['lat'], city2['lon']
        )
        
        return {
            'euclidean_distance': euclidean_dist,
            'haversine_distance_km': haversine_dist,
            'distance_category': 'short' if haversine_dist < 100 else 'medium' if haversine_dist < 300 else 'long'
        }
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance in kilometers"""
        
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _calculate_population_features(self, city1: pd.Series, city2: pd.Series) -> Dict:
        """Calculate population-based features"""
        
        pop1 = city1.get('population', 0)
        pop2 = city2.get('population', 0)
        
        return {
            'population_sum': pop1 + pop2,
            'population_mean': (pop1 + pop2) / 2,
            'population_ratio': max(pop1, pop2) / (min(pop1, pop2) + 1),
            'population_product': pop1 * pop2,
            'population_difference': abs(pop1 - pop2)
        }
    
    def _calculate_economic_features(self, city1: pd.Series, city2: pd.Series) -> Dict:
        """Calculate economic features"""
        
        gdp1 = city1.get('gdp_per_capita', 0)
        gdp2 = city2.get('gdp_per_capita', 0)
        
        econ1 = city1.get('economic_activity', 0)
        econ2 = city2.get('economic_activity', 0)
        
        return {
            'gdp_sum': gdp1 + gdp2,
            'gdp_mean': (gdp1 + gdp2) / 2,
            'gdp_ratio': max(gdp1, gdp2) / (min(gdp1, gdp2) + 1),
            'economic_activity_sum': econ1 + econ2,
            'economic_activity_mean': (econ1 + econ2) / 2,
            'economic_complementarity': abs(econ1 - econ2) / (econ1 + econ2 + 1)
        }
    
    def _calculate_geographic_features(self, city1: pd.Series, city2: pd.Series) -> Dict:
        """Calculate geographic features"""
        
        lat_diff = abs(city1['lat'] - city2['lat'])
        lon_diff = abs(city1['lon'] - city2['lon'])
        
        return {
            'latitude_difference': lat_diff,
            'longitude_difference': lon_diff,
            'coordinate_similarity': 1 / (lat_diff + lon_diff + 0.01),
            'same_region': (
                city1.get('region', '') == city2.get('region', '') and 
                city1.get('region', '') != ''
            )
        }
    
    def _calculate_connectivity_features(self, city1: pd.Series, city2: pd.Series) -> Dict:
        """Calculate connectivity features"""
        
        conn1 = city1.get('connectivity_score', 0)
        conn2 = city2.get('connectivity_score', 0)
        
        return {
            'connectivity_sum': conn1 + conn2,
            'connectivity_mean': (conn1 + conn2) / 2,
            'connectivity_ratio': max(conn1, conn2) / (min(conn1, conn2) + 0.01)
        }
    
    def _calculate_demand_features(self, city1: pd.Series, city2: pd.Series) -> Dict:
        """Calculate demand prediction features using gravity model"""
        
        pop1 = city1.get('population', 0)
        pop2 = city2.get('population', 0)
        
        # Distance in km
        distance = self._haversine_distance(
            city1['lat'], city1['lon'], city2['lat'], city2['lon']
        )
        
        # Gravity model: demand = (pop1 * pop2) / distance^2
        if distance > 0:
            gravity_demand = (pop1 * pop2) / (distance ** 2)
        else:
            gravity_demand = 0
        
        # Modified gravity model with economic factors
        gdp1 = city1.get('gdp_per_capita', 30000)
        gdp2 = city2.get('gdp_per_capita', 30000)
        
        economic_factor = (gdp1 + gdp2) / 60000  # Normalize
        
        modified_demand = gravity_demand * economic_factor
        
        return {
            'gravity_demand': gravity_demand,
            'modified_gravity_demand': modified_demand,
            'demand_intensity': modified_demand / (distance + 1),
            'distance_decay': np.exp(-distance / 100)  # Exponential decay
        }
    
    def create_network_features(self, network_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Create features from existing railway network"""
        
        if network_gdf.empty:
            return pd.DataFrame()
        
        network_features = []
        
        for idx, segment in network_gdf.iterrows():
            features = {
                'segment_id': idx,
                'length_km': segment.geometry.length * 111,  # Convert to km
                'railway_type': segment.get('railway_type', 'unknown'),
                'electrified': segment.get('electrified', 'no') == 'yes',
                'usage': segment.get('usage', 'unknown'),
                'max_speed': self._extract_max_speed(segment.get('maxspeed', '')),
                'gauge': self._extract_gauge(segment.get('gauge', '')),
                'operator': segment.get('operator', 'unknown')
            }
            
            # Add geometric features
            features.update(self._calculate_segment_geometry_features(segment.geometry))
            
            # Add connectivity features
            features.update(self._calculate_segment_connectivity_features(segment, network_gdf))
            
            network_features.append(features)
        
        return pd.DataFrame(network_features)
    
    def _extract_max_speed(self, speed_str: str) -> float:
        """Extract maximum speed from string"""
        
        if not speed_str:
            return 0
        
        # Extract numeric value
        import re
        match = re.search(r'(\d+)', str(speed_str))
        if match:
            return float(match.group(1))
        
        return 0
    
    def _extract_gauge(self, gauge_str: str) -> float:
        """Extract gauge from string"""
        
        if not gauge_str:
            return 1435  # Standard gauge
        
        # Extract numeric value
        import re
        match = re.search(r'(\d+)', str(gauge_str))
        if match:
            return float(match.group(1))
        
        return 1435
    
    def _calculate_segment_geometry_features(self, geometry: LineString) -> Dict:
        """Calculate geometric features for a railway segment"""
        
        coords = list(geometry.coords)
        
        # Basic geometry
        features = {
            'start_lat': coords[0][1],
            'start_lon': coords[0][0],
            'end_lat': coords[-1][1],
            'end_lon': coords[-1][0],
            'point_count': len(coords)
        }
        
        # Calculate tortuosity (actual length / straight line distance)
        if len(coords) >= 2:
            straight_line_dist = np.sqrt(
                (coords[-1][0] - coords[0][0])**2 + 
                (coords[-1][1] - coords[0][1])**2
            )
            actual_length = geometry.length
            
            features['tortuosity'] = actual_length / (straight_line_dist + 1e-6)
        else:
            features['tortuosity'] = 1.0
        
        return features
    
    def _calculate_segment_connectivity_features(self, segment: pd.Series, 
                                               network_gdf: gpd.GeoDataFrame) -> Dict:
        """Calculate connectivity features for a railway segment"""
        
        # Count nearby segments
        buffer_distance = 0.01  # Roughly 1km
        segment_buffer = segment.geometry.buffer(buffer_distance)
        
        nearby_segments = network_gdf[network_gdf.geometry.intersects(segment_buffer)]
        
        return {
            'nearby_segments': len(nearby_segments) - 1,  # Exclude self
            'connectivity_degree': len(nearby_segments) - 1
        }
    
    def save_features(self, features_df: pd.DataFrame, output_path: str, 
                     feature_type: str = "general"):
        """Save engineered features to file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        features_df.to_csv(output_file, index=False)
        
        # Save feature metadata
        metadata = {
            'feature_count': len(features_df.columns),
            'record_count': len(features_df),
            'feature_type': feature_type,
            'feature_names': list(features_df.columns),
            'feature_types': features_df.dtypes.to_dict()
        }
        
        metadata_file = output_file.parent / f"{output_file.stem}_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(features_df)} features to {output_file}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create feature engineer
    engineer = FeatureEngineer()
    
    # Test with sample data
    sample_cities = pd.DataFrame({
        'city': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne'],
        'lat': [52.5200, 48.1351, 53.5511, 50.1109, 50.9375],
        'lon': [13.4050, 11.5820, 9.9937, 8.6821, 6.9603],
        'population': [3669491, 1471508, 1899160, 753056, 1085664],
        'gdp_per_capita': [42000, 45000, 44000, 48000, 43000],
        'economic_activity': [8.5, 8.2, 7.8, 8.8, 7.5],
        'tourism_index': [8.2, 7.5, 6.8, 6.2, 6.5]
    })
    
    # Create city features
    city_features = engineer.create_city_features(sample_cities)
    
    print("=== City Features (sample) ===")
    print(city_features[['city', 'population', 'log_population', 'connectivity_score', 'spatial_cluster']].head())
    
    # Create route features
    city_pairs = [('Berlin', 'Munich'), ('Hamburg', 'Frankfurt'), ('Cologne', 'Berlin')]
    route_features = engineer.create_route_features(city_pairs, city_features)
    
    print("\n=== Route Features ===")
    print(route_features[['route_id', 'haversine_distance_km', 'gravity_demand', 'population_sum']].head())
    
    # Save features
    engineer.save_features(city_features, "data/processed/features/city_features.csv", "city")
    engineer.save_features(route_features, "data/processed/features/route_features.csv", "route")