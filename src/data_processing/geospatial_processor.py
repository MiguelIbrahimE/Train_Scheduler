import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString
from shapely.ops import unary_union, nearest_points, transform
from shapely.spatial import voronoi_diagram
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
import requests
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pyproj
from pyproj import Transformer
import folium
from scipy.spatial import Voronoi, voronoi_plot_2d
import networkx as nx

class GeospatialProcessor:
    """Handle geospatial data processing for railway route optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common CRS transformations
        self.wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 (lat/lon)
        self.web_mercator = pyproj.CRS('EPSG:3857')  # Web Mercator (meters)
        
        # European projections for better distance calculations
        self.european_projections = {
            'germany': pyproj.CRS('EPSG:31467'),  # DHDN / 3-degree Gauss-Kruger zone 3
            'france': pyproj.CRS('EPSG:2154'),    # RGF93 / Lambert-93
            'switzerland': pyproj.CRS('EPSG:2056'), # CH1903+ / LV95
            'austria': pyproj.CRS('EPSG:31256'),   # MGI / Austria GK Central
            'netherlands': pyproj.CRS('EPSG:28992'), # Amersfoort / RD New
            'denmark': pyproj.CRS('EPSG:25832'),   # ETRS89 / UTM zone 32N
            'sweden': pyproj.CRS('EPSG:3006'),     # SWEREF99 TM
            'japan': pyproj.CRS('EPSG:6677')      # JGD2011 / Japan Plane Rectangular CS IX
        }
    
    def reproject_to_metric(self, gdf: gpd.GeoDataFrame, country: str = None) -> gpd.GeoDataFrame:
        """Reproject GeoDataFrame to appropriate metric coordinate system"""
        
        if gdf.empty:
            return gdf
        
        # Select appropriate projection
        if country and country in self.european_projections:
            target_crs = self.european_projections[country]
        else:
            # Default to Web Mercator for global use
            target_crs = self.web_mercator
        
        # Reproject if not already in target CRS
        if gdf.crs != target_crs:
            gdf_reprojected = gdf.to_crs(target_crs)
            self.logger.info(f"Reprojected from {gdf.crs} to {target_crs}")
            return gdf_reprojected
        
        return gdf
    
    def calculate_accurate_distances(self, gdf: gpd.GeoDataFrame, country: str = None) -> gpd.GeoDataFrame:
        """Calculate accurate distances using appropriate projections"""
        
        gdf_metric = self.reproject_to_metric(gdf, country)
        
        # Calculate length for LineString geometries
        if not gdf_metric.empty and gdf_metric.geometry.geom_type.iloc[0] in ['LineString', 'MultiLineString']:
            gdf_metric['length_m'] = gdf_metric.geometry.length
            gdf_metric['length_km'] = gdf_metric['length_m'] / 1000
        
        # Calculate area for Polygon geometries
        elif not gdf_metric.empty and gdf_metric.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
            gdf_metric['area_m2'] = gdf_metric.geometry.area
            gdf_metric['area_km2'] = gdf_metric['area_m2'] / 1000000
        
        return gdf_metric
    
    def create_buffer_analysis(self, gdf: gpd.GeoDataFrame, buffer_distance_km: float, 
                              country: str = None) -> gpd.GeoDataFrame:
        """Create buffer analysis around geometries"""
        
        # Reproject to metric system
        gdf_metric = self.reproject_to_metric(gdf, country)
        
        if gdf_metric.empty:
            return gdf_metric
        
        # Create buffer (distance in meters)
        buffer_distance_m = buffer_distance_km * 1000
        gdf_metric['buffer'] = gdf_metric.geometry.buffer(buffer_distance_m)
        
        # Calculate buffer area
        gdf_metric['buffer_area_km2'] = gdf_metric['buffer'].area / 1000000
        
        return gdf_metric
    
    def find_nearest_features(self, source_gdf: gpd.GeoDataFrame, 
                            target_gdf: gpd.GeoDataFrame, 
                            country: str = None) -> gpd.GeoDataFrame:
        """Find nearest features between two GeoDataFrames"""
        
        if source_gdf.empty or target_gdf.empty:
            return source_gdf
        
        # Reproject both to metric system
        source_metric = self.reproject_to_metric(source_gdf, country)
        target_metric = self.reproject_to_metric(target_gdf, country)
        
        # Find nearest features
        nearest_data = []
        
        for idx, source_row in source_metric.iterrows():
            distances = target_metric.geometry.distance(source_row.geometry)
            nearest_idx = distances.idxmin()
            nearest_distance = distances.iloc[nearest_idx]
            
            nearest_data.append({
                'source_idx': idx,
                'nearest_target_idx': nearest_idx,
                'distance_m': nearest_distance,
                'distance_km': nearest_distance / 1000
            })
        
        # Add nearest information to source GeoDataFrame
        nearest_df = pd.DataFrame(nearest_data)
        source_with_nearest = source_gdf.copy()
        source_with_nearest['nearest_target_idx'] = nearest_df['nearest_target_idx']
        source_with_nearest['nearest_distance_km'] = nearest_df['distance_km']
        
        return source_with_nearest
    
    def create_voronoi_regions(self, points_gdf: gpd.GeoDataFrame, 
                              boundary_gdf: gpd.GeoDataFrame = None,
                              country: str = None) -> gpd.GeoDataFrame:
        """Create Voronoi regions from points"""
        
        if points_gdf.empty:
            return gpd.GeoDataFrame()
        
        # Reproject to metric system
        points_metric = self.reproject_to_metric(points_gdf, country)
        
        # Extract coordinates
        coords = np.array([[geom.x, geom.y] for geom in points_metric.geometry])
        
        # Create Voronoi diagram
        vor = Voronoi(coords)
        
        # Create polygons from Voronoi regions
        polygons = []
        for point_idx, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            
            if len(region) > 0 and -1 not in region:
                # Valid finite region
                polygon_coords = [vor.vertices[i] for i in region]
                if len(polygon_coords) >= 3:
                    polygons.append(Polygon(polygon_coords))
                else:
                    polygons.append(None)
            else:
                polygons.append(None)
        
        # Create GeoDataFrame
        voronoi_gdf = gpd.GeoDataFrame(
            points_metric.drop('geometry', axis=1),
            geometry=polygons,
            crs=points_metric.crs
        )
        
        # Remove invalid geometries
        voronoi_gdf = voronoi_gdf[voronoi_gdf.geometry.notna()]
        
        # Clip to boundary if provided
        if boundary_gdf is not None and not boundary_gdf.empty:
            boundary_metric = self.reproject_to_metric(boundary_gdf, country)
            boundary_union = unary_union(boundary_metric.geometry)
            voronoi_gdf['geometry'] = voronoi_gdf.geometry.intersection(boundary_union)
        
        return voronoi_gdf
    
    def create_network_graph(self, network_gdf: gpd.GeoDataFrame, 
                           stations_gdf: gpd.GeoDataFrame = None,
                           country: str = None) -> nx.Graph:
        """Create network graph from railway data"""
        
        if network_gdf.empty:
            return nx.Graph()
        
        # Reproject to metric system
        network_metric = self.reproject_to_metric(network_gdf, country)
        
        G = nx.Graph()
        
        # Add railway segments as edges
        for idx, segment in network_metric.iterrows():
            if segment.geometry.geom_type == 'LineString':
                coords = list(segment.geometry.coords)
                
                # Add nodes for start and end points
                start_node = f"coord_{coords[0][0]:.0f}_{coords[0][1]:.0f}"
                end_node = f"coord_{coords[-1][0]:.0f}_{coords[-1][1]:.0f}"
                
                G.add_node(start_node, pos=coords[0])
                G.add_node(end_node, pos=coords[-1])
                
                # Add edge with attributes
                edge_attrs = {
                    'length_m': segment.geometry.length,
                    'length_km': segment.geometry.length / 1000,
                    'railway_type': segment.get('railway_type', 'unknown'),
                    'segment_id': idx
                }
                
                G.add_edge(start_node, end_node, **edge_attrs)
        
        # Add station nodes if provided
        if stations_gdf is not None and not stations_gdf.empty:
            stations_metric = self.reproject_to_metric(stations_gdf, country)
            
            for idx, station in stations_metric.iterrows():
                station_node = f"station_{idx}"
                G.add_node(
                    station_node,
                    pos=(station.geometry.x, station.geometry.y),
                    type='station',
                    name=station.get('name', 'Unknown')
                )
                
                # Connect station to nearest railway segment
                # (simplified - in practice, would need more sophisticated connection logic)
                distances = network_metric.geometry.distance(station.geometry)
                if not distances.empty:
                    nearest_segment_idx = distances.idxmin()
                    nearest_segment = network_metric.iloc[nearest_segment_idx]
                    
                    # Find nearest point on segment
                    nearest_point = nearest_points(station.geometry, nearest_segment.geometry)[1]
                    
                    # Create connection node
                    conn_node = f"conn_{idx}"
                    G.add_node(conn_node, pos=(nearest_point.x, nearest_point.y))
                    
                    # Add edge from station to connection point
                    G.add_edge(station_node, conn_node, 
                              length_m=station.geometry.distance(nearest_point),
                              type='station_connection')
        
        return G
    
    def calculate_network_metrics(self, network_gdf: gpd.GeoDataFrame, 
                                country: str = None) -> Dict:
        """Calculate network connectivity metrics"""
        
        if network_gdf.empty:
            return {}
        
        # Create network graph
        G = self.create_network_graph(network_gdf, country=country)
        
        # Basic network metrics
        metrics = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'total_length_km': sum(data.get('length_km', 0) for _, _, data in G.edges(data=True)),
            'connected_components': nx.number_connected_components(G),
            'is_connected': nx.is_connected(G)
        }
        
        if G.number_of_nodes() > 0:
            # Centrality measures
            try:
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                closeness_centrality = nx.closeness_centrality(G)
                
                metrics.update({
                    'avg_degree_centrality': np.mean(list(degree_centrality.values())),
                    'avg_betweenness_centrality': np.mean(list(betweenness_centrality.values())),
                    'avg_closeness_centrality': np.mean(list(closeness_centrality.values()))
                })
            except:
                pass
            
            # Network efficiency
            try:
                if nx.is_connected(G):
                    efficiency = nx.global_efficiency(G)
                    metrics['network_efficiency'] = efficiency
            except:
                pass
        
        return metrics
    
    def create_accessibility_analysis(self, origins_gdf: gpd.GeoDataFrame,
                                    destinations_gdf: gpd.GeoDataFrame,
                                    network_gdf: gpd.GeoDataFrame = None,
                                    max_distance_km: float = 50,
                                    country: str = None) -> gpd.GeoDataFrame:
        """Create accessibility analysis between origins and destinations"""
        
        if origins_gdf.empty or destinations_gdf.empty:
            return gpd.GeoDataFrame()
        
        # Reproject to metric system
        origins_metric = self.reproject_to_metric(origins_gdf, country)
        destinations_metric = self.reproject_to_metric(destinations_gdf, country)
        
        accessibility_results = []
        
        for origin_idx, origin in origins_metric.iterrows():
            # Calculate distances to all destinations
            distances = destinations_metric.geometry.distance(origin.geometry)
            
            # Filter by maximum distance
            max_distance_m = max_distance_km * 1000
            accessible_destinations = destinations_metric[distances <= max_distance_m]
            
            # Calculate accessibility metrics
            accessibility_metrics = {
                'origin_id': origin_idx,
                'accessible_destinations': len(accessible_destinations),
                'total_destinations': len(destinations_metric),
                'accessibility_ratio': len(accessible_destinations) / len(destinations_metric),
                'mean_distance_km': distances.mean() / 1000,
                'min_distance_km': distances.min() / 1000,
                'max_distance_km': distances.max() / 1000
            }
            
            # Add population-weighted accessibility if population data available
            if 'population' in accessible_destinations.columns:
                total_accessible_pop = accessible_destinations['population'].sum()
                accessibility_metrics['accessible_population'] = total_accessible_pop
                
                # Gravity-based accessibility
                gravity_accessibility = sum(
                    pop / (dist/1000 + 1)**2 
                    for pop, dist in zip(accessible_destinations['population'], 
                                       distances[distances <= max_distance_m])
                )
                accessibility_metrics['gravity_accessibility'] = gravity_accessibility
            
            accessibility_results.append(accessibility_metrics)
        
        # Create results GeoDataFrame
        accessibility_df = pd.DataFrame(accessibility_results)
        accessibility_gdf = origins_gdf.copy()
        
        for col in accessibility_df.columns:
            if col != 'origin_id':
                accessibility_gdf[col] = accessibility_df[col].values
        
        return accessibility_gdf
    
    def create_corridor_analysis(self, start_point: Point, end_point: Point,
                               corridor_width_km: float = 10,
                               country: str = None) -> Polygon:
        """Create corridor polygon between two points"""
        
        # Create GeoDataFrame for points
        points_gdf = gpd.GeoDataFrame(
            geometry=[start_point, end_point],
            crs=self.wgs84
        )
        
        # Reproject to metric system
        points_metric = self.reproject_to_metric(points_gdf, country)
        
        # Create line between points
        start_metric = points_metric.geometry.iloc[0]
        end_metric = points_metric.geometry.iloc[1]
        
        corridor_line = LineString([start_metric, end_metric])
        
        # Create buffer around line
        corridor_width_m = corridor_width_km * 1000
        corridor_polygon = corridor_line.buffer(corridor_width_m / 2)
        
        return corridor_polygon
    
    def spatial_join_analysis(self, left_gdf: gpd.GeoDataFrame, 
                            right_gdf: gpd.GeoDataFrame,
                            join_type: str = 'intersects',
                            country: str = None) -> gpd.GeoDataFrame:
        """Perform spatial join analysis"""
        
        if left_gdf.empty or right_gdf.empty:
            return left_gdf
        
        # Reproject both to same CRS
        left_metric = self.reproject_to_metric(left_gdf, country)
        right_metric = self.reproject_to_metric(right_gdf, country)
        
        # Perform spatial join
        joined_gdf = gpd.sjoin(left_metric, right_metric, how='left', predicate=join_type)
        
        return joined_gdf
    
    def create_density_analysis(self, points_gdf: gpd.GeoDataFrame,
                              grid_size_km: float = 1,
                              country: str = None) -> gpd.GeoDataFrame:
        """Create density analysis using regular grid"""
        
        if points_gdf.empty:
            return gpd.GeoDataFrame()
        
        # Reproject to metric system
        points_metric = self.reproject_to_metric(points_gdf, country)
        
        # Get bounds
        bounds = points_metric.total_bounds
        
        # Create regular grid
        grid_size_m = grid_size_km * 1000
        
        x_coords = np.arange(bounds[0], bounds[2], grid_size_m)
        y_coords = np.arange(bounds[1], bounds[3], grid_size_m)
        
        # Create grid polygons
        grid_polygons = []
        for x in x_coords:
            for y in y_coords:
                grid_polygon = Polygon([
                    (x, y),
                    (x + grid_size_m, y),
                    (x + grid_size_m, y + grid_size_m),
                    (x, y + grid_size_m)
                ])
                grid_polygons.append(grid_polygon)
        
        # Create grid GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(
            geometry=grid_polygons,
            crs=points_metric.crs
        )
        
        # Count points in each grid cell
        joined = gpd.sjoin(points_metric, grid_gdf, how='right', predicate='within')
        density_counts = joined.groupby('index_right').size().reset_index(name='point_count')
        
        # Add density information to grid
        grid_gdf['point_count'] = 0
        grid_gdf.loc[density_counts['index_right'], 'point_count'] = density_counts['point_count']
        
        # Calculate density (points per km²)
        grid_gdf['density_per_km2'] = grid_gdf['point_count'] / (grid_size_km ** 2)
        
        return grid_gdf
    
    def export_to_formats(self, gdf: gpd.GeoDataFrame, output_path: str, 
                         formats: List[str] = ['geojson', 'shapefile']):
        """Export GeoDataFrame to multiple formats"""
        
        if gdf.empty:
            return
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure WGS84 for export
        if gdf.crs != self.wgs84:
            gdf_export = gdf.to_crs(self.wgs84)
        else:
            gdf_export = gdf
        
        for format_type in formats:
            if format_type.lower() == 'geojson':
                export_path = output_file.parent / f"{output_file.stem}.geojson"
                gdf_export.to_file(export_path, driver='GeoJSON')
                
            elif format_type.lower() == 'shapefile':
                export_path = output_file.parent / f"{output_file.stem}.shp"
                gdf_export.to_file(export_path, driver='ESRI Shapefile')
                
            elif format_type.lower() == 'kml':
                export_path = output_file.parent / f"{output_file.stem}.kml"
                gdf_export.to_file(export_path, driver='KML')
                
            elif format_type.lower() == 'gpkg':
                export_path = output_file.parent / f"{output_file.stem}.gpkg"
                gdf_export.to_file(export_path, driver='GPKG')
        
        self.logger.info(f"Exported to {len(formats)} formats: {output_file.parent}")
    
    def create_interactive_map(self, gdf_list: List[gpd.GeoDataFrame], 
                             names: List[str] = None,
                             colors: List[str] = None,
                             output_path: str = None) -> folium.Map:
        """Create interactive map with multiple layers"""
        
        # Filter out empty GeoDataFrames
        valid_gdfs = [(gdf, name, color) for gdf, name, color in 
                     zip(gdf_list, names or [f"Layer {i}" for i in range(len(gdf_list))], 
                         colors or ['red', 'blue', 'green', 'orange', 'purple']) 
                     if not gdf.empty]
        
        if not valid_gdfs:
            self.logger.warning("No valid GeoDataFrames for mapping")
            return folium.Map()
        
        # Calculate center point
        all_bounds = []
        for gdf, _, _ in valid_gdfs:
            # Ensure WGS84
            if gdf.crs != self.wgs84:
                gdf_wgs84 = gdf.to_crs(self.wgs84)
            else:
                gdf_wgs84 = gdf
            all_bounds.append(gdf_wgs84.total_bounds)
        
        combined_bounds = np.array(all_bounds)
        center_lat = (combined_bounds[:, 1].min() + combined_bounds[:, 3].max()) / 2
        center_lon = (combined_bounds[:, 0].min() + combined_bounds[:, 2].max()) / 2
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        # Add layers
        for gdf, name, color in valid_gdfs:
            # Ensure WGS84
            if gdf.crs != self.wgs84:
                gdf_wgs84 = gdf.to_crs(self.wgs84)
            else:
                gdf_wgs84 = gdf
            
            # Add to map based on geometry type
            if not gdf_wgs84.empty:
                geom_type = gdf_wgs84.geometry.geom_type.iloc[0]
                
                if geom_type == 'Point':
                    for _, row in gdf_wgs84.iterrows():
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=5,
                            color=color,
                            fill=True,
                            popup=f"{name}: {row.get('name', 'Unknown')}"
                        ).add_to(m)
                
                elif geom_type in ['LineString', 'MultiLineString']:
                    for _, row in gdf_wgs84.iterrows():
                        coords = [[lat, lon] for lon, lat in row.geometry.coords]
                        folium.PolyLine(
                            locations=coords,
                            color=color,
                            weight=3,
                            opacity=0.7,
                            popup=f"{name}: {row.get('name', 'Unknown')}"
                        ).add_to(m)
                
                elif geom_type in ['Polygon', 'MultiPolygon']:
                    for _, row in gdf_wgs84.iterrows():
                        if row.geometry.geom_type == 'Polygon':
                            coords = [[lat, lon] for lon, lat in row.geometry.exterior.coords]
                            folium.Polygon(
                                locations=coords,
                                color=color,
                                weight=2,
                                opacity=0.7,
                                fill=True,
                                fillOpacity=0.3,
                                popup=f"{name}: {row.get('name', 'Unknown')}"
                            ).add_to(m)
        
        # Save map if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            m.save(str(output_file))
            self.logger.info(f"Interactive map saved to {output_file}")
        
        return m

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create processor
    processor = GeospatialProcessor()
    
    # Test with sample data
    sample_points = gpd.GeoDataFrame(
        {
            'name': ['Berlin', 'Munich', 'Hamburg'],
            'population': [3669491, 1471508, 1899160]
        },
        geometry=[
            Point(13.4050, 52.5200),
            Point(11.5820, 48.1351),
            Point(9.9937, 53.5511)
        ],
        crs='EPSG:4326'
    )
    
    sample_lines = gpd.GeoDataFrame(
        {
            'route': ['Berlin-Munich', 'Berlin-Hamburg'],
            'type': ['ICE', 'IC']
        },
        geometry=[
            LineString([(13.4050, 52.5200), (11.5820, 48.1351)]),
            LineString([(13.4050, 52.5200), (9.9937, 53.5511)])
        ],
        crs='EPSG:4326'
    )
    
    # Test reprojection
    points_metric = processor.reproject_to_metric(sample_points, 'germany')
    print(f"Original CRS: {sample_points.crs}")
    print(f"Reprojected CRS: {points_metric.crs}")
    
    # Test distance calculation
    lines_with_distance = processor.calculate_accurate_distances(sample_lines, 'germany')
    print(f"Route distances: {lines_with_distance[['route', 'length_km']].to_dict()}")
    
    # Test accessibility analysis
    accessibility = processor.create_accessibility_analysis(
        sample_points, sample_points, max_distance_km=100, country='germany'
    )
    print(f"Accessibility results: {accessibility[['name', 'accessible_destinations', 'mean_distance_km']].to_dict()}")
    
    # Test network metrics
    network_metrics = processor.calculate_network_metrics(sample_lines, 'germany')
    print(f"Network metrics: {network_metrics}")
    
    # Create interactive map
    interactive_map = processor.create_interactive_map(
        [sample_points, sample_lines],
        ['Cities', 'Routes'],
        ['red', 'blue'],
        'data/output/visualizations/geospatial_test_map.html'
    )
    
    print("Geospatial processor test completed successfully!")