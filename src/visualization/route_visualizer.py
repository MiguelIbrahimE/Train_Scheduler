import folium
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import numpy as np

class RouteVisualizer:
    """Create interactive visualizations for railway routes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Color schemes
        self.route_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        self.train_type_colors = {
            'high_speed': '#FF0000',
            'intercity': '#0066CC',
            'regional': '#00AA00',
            'commuter': '#FF8800',
            'rail': '#666666'
        }
    
    def create_interactive_map(self, cities: pd.DataFrame, optimized_routes: Dict,
                             existing_network: gpd.GeoDataFrame = None, 
                             country: str = "Unknown") -> str:
        """Create an interactive map showing cities, existing network, and optimized routes"""
        
        # Calculate map center
        center_lat = cities['lat'].mean()
        center_lon = cities['lon'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )
        
        # Add existing railway network
        if existing_network is not None and not existing_network.empty:
            self._add_existing_network(m, existing_network)
        
        # Add cities
        self._add_cities(m, cities)
        
        # Add optimized routes
        self._add_optimized_routes(m, optimized_routes, cities)
        
        # Add legend
        self._add_legend(m)
        
        # Save map
        output_dir = Path("data/output/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        map_file = output_dir / f"{country}_railway_map.html"
        m.save(str(map_file))
        
        self.logger.info(f"Interactive map saved to {map_file}")
        return str(map_file)
    
    def _add_existing_network(self, m: folium.Map, network: gpd.GeoDataFrame):
        """Add existing railway network to map"""
        
        # Group by railway type
        railway_types = network['railway_type'].unique() if 'railway_type' in network.columns else ['rail']
        
        for rail_type in railway_types:
            if rail_type in network['railway_type'].values:
                subset = network[network['railway_type'] == rail_type]
            else:
                subset = network
            
            color = self.train_type_colors.get(rail_type, '#666666')
            
            # Add each segment
            for _, segment in subset.iterrows():
                if hasattr(segment.geometry, 'coords'):
                    coords = [[lat, lon] for lon, lat in segment.geometry.coords]
                    
                    # Create popup info
                    popup_text = f"""
                    <b>Existing Railway</b><br>
                    Type: {rail_type}<br>
                    Operator: {segment.get('operator', 'Unknown')}<br>
                    Electrified: {segment.get('electrified', 'Unknown')}<br>
                    Max Speed: {segment.get('maxspeed', 'Unknown')}
                    """
                    
                    folium.PolyLine(
                        locations=coords,
                        color=color,
                        weight=2,
                        opacity=0.7,
                        popup=folium.Popup(popup_text, max_width=300)
                    ).add_to(m)
    
    def _add_cities(self, m: folium.Map, cities: pd.DataFrame):
        """Add cities to map"""
        
        # Normalize population for marker size
        max_pop = cities['population'].max()
        min_pop = cities['population'].min()
        
        for _, city in cities.iterrows():
            # Calculate marker size based on population
            pop_ratio = (city['population'] - min_pop) / (max_pop - min_pop)
            marker_size = 5 + (pop_ratio * 15)  # Size between 5 and 20
            
            # Create popup info
            popup_text = f"""
            <b>{city['city']}</b><br>
            Population: {city['population']:,}<br>
            Coordinates: {city['lat']:.4f}, {city['lon']:.4f}<br>
            GDP per capita: ${city.get('gdp_per_capita', 'Unknown'):,}<br>
            Tourism Index: {city.get('tourism_index', 'Unknown'):.2f}
            """
            
            folium.CircleMarker(
                location=[city['lat'], city['lon']],
                radius=marker_size,
                popup=folium.Popup(popup_text, max_width=300),
                color='black',
                weight=2,
                fill=True,
                fillColor='yellow',
                fillOpacity=0.7
            ).add_to(m)
            
            # Add city name label
            folium.Marker(
                location=[city['lat'], city['lon']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12px; font-weight: bold; color: black;">{city["city"]}</div>',
                    icon_size=(100, 20),
                    icon_anchor=(50, 10)
                )
            ).add_to(m)
    
    def _add_optimized_routes(self, m: folium.Map, routes: Dict, cities: pd.DataFrame):
        """Add optimized routes to map"""
        
        color_idx = 0
        
        for route_id, route_data in routes.items():
            if not route_data['path']:
                continue
            
            # Get route coordinates
            coords = []
            for city_name in route_data['path']:
                city_data = cities[cities['city'] == city_name]
                if not city_data.empty:
                    coords.append([city_data.iloc[0]['lat'], city_data.iloc[0]['lon']])
            
            if len(coords) < 2:
                continue
            
            # Select color
            color = self.route_colors[color_idx % len(self.route_colors)]
            color_idx += 1
            
            # Create popup info
            total_distance = sum(seg['distance_km'] for seg in route_data['segments'])
            popup_text = f"""
            <b>Optimized Route</b><br>
            Route ID: {route_id}<br>
            Path: {' → '.join(route_data['path'])}<br>
            Total Distance: {total_distance:.1f} km<br>
            Total Cost: {route_data['total_cost']:.2f}<br>
            Segments: {len(route_data['segments'])}
            """
            
            # Add route line
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=4,
                opacity=0.8,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
            
            # Add route markers at start and end
            folium.CircleMarker(
                location=coords[0],
                radius=8,
                color=color,
                weight=3,
                fill=True,
                fillColor='white',
                fillOpacity=1,
                popup=f"Start: {route_data['path'][0]}"
            ).add_to(m)
            
            folium.CircleMarker(
                location=coords[-1],
                radius=8,
                color=color,
                weight=3,
                fill=True,
                fillColor='red',
                fillOpacity=1,
                popup=f"End: {route_data['path'][-1]}"
            ).add_to(m)
    
    def _add_legend(self, m: folium.Map):
        """Add legend to map"""
        
        legend_html = """
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Railway Map Legend</h4>
        
        <p><b>Cities:</b></p>
        <i class="fa fa-circle" style="color:yellow"></i> City (size = population)<br>
        
        <p><b>Existing Network:</b></p>
        <i class="fa fa-minus" style="color:#FF0000"></i> High Speed<br>
        <i class="fa fa-minus" style="color:#0066CC"></i> Intercity<br>
        <i class="fa fa-minus" style="color:#00AA00"></i> Regional<br>
        <i class="fa fa-minus" style="color:#FF8800"></i> Commuter<br>
        
        <p><b>Optimized Routes:</b></p>
        <i class="fa fa-minus" style="color:#FF6B6B; font-weight:bold"></i> New Route<br>
        <i class="fa fa-circle" style="color:white; border: 2px solid black"></i> Start<br>
        <i class="fa fa-circle" style="color:red"></i> End<br>
        
        </div>
        """
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def create_route_comparison_chart(self, optimization_reports: Dict[str, pd.DataFrame], 
                                    output_path: str = "data/output/visualizations/route_comparison.html"):
        """Create comparison chart for routes across countries"""
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Combine all reports
            all_reports = []
            for country, report in optimization_reports.items():
                report_copy = report.copy()
                report_copy['country'] = country
                all_reports.append(report_copy)
            
            if not all_reports:
                self.logger.warning("No optimization reports to visualize")
                return
            
            combined_report = pd.concat(all_reports, ignore_index=True)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cost per Country', 'Distance vs Efficiency', 
                               'Population Served', 'Terrain Cost Analysis'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Cost per country
            country_costs = combined_report.groupby('country')['total_cost'].mean()
            fig.add_trace(
                go.Bar(x=country_costs.index, y=country_costs.values, 
                       name='Avg Cost', marker_color='lightblue'),
                row=1, col=1
            )
            
            # 2. Distance vs Efficiency scatter
            fig.add_trace(
                go.Scatter(
                    x=combined_report['total_distance_km'],
                    y=combined_report['route_efficiency'],
                    mode='markers',
                    marker=dict(
                        size=combined_report['population_served']/10000,
                        color=combined_report['total_cost'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=combined_report['country'],
                    name='Routes'
                ),
                row=1, col=2
            )
            
            # 3. Population served by country
            country_pop = combined_report.groupby('country')['population_served'].sum()
            fig.add_trace(
                go.Bar(x=country_pop.index, y=country_pop.values, 
                       name='Population Served', marker_color='lightgreen'),
                row=2, col=1
            )
            
            # 4. Terrain cost analysis
            if 'terrain_cost' in combined_report.columns:
                country_terrain = combined_report.groupby('country')['terrain_cost'].mean()
                fig.add_trace(
                    go.Bar(x=country_terrain.index, y=country_terrain.values, 
                           name='Avg Terrain Cost', marker_color='orange'),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text="Railway Route Optimization Analysis",
                showlegend=False,
                height=800
            )
            
            # Save chart
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_file))
            
            self.logger.info(f"Comparison chart saved to {output_file}")
            
        except ImportError:
            self.logger.warning("Plotly not available, skipping comparison chart")
        except Exception as e:
            self.logger.error(f"Error creating comparison chart: {e}")
    
    def create_network_analysis_chart(self, existing_networks: Dict[str, gpd.GeoDataFrame],
                                    output_path: str = "data/output/visualizations/network_analysis.html"):
        """Create analysis chart for existing railway networks"""
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Analyze networks
            network_stats = []
            for country, network in existing_networks.items():
                if network.empty:
                    continue
                
                stats = {
                    'country': country,
                    'total_length': network.geometry.length.sum() * 111,  # Convert to km
                    'segment_count': len(network),
                    'avg_segment_length': network.geometry.length.mean() * 111,
                    'electrified_ratio': len(network[network.get('electrified', '') == 'yes']) / len(network) if len(network) > 0 else 0,
                    'high_speed_ratio': len(network[network.get('railway_type', '') == 'high_speed']) / len(network) if len(network) > 0 else 0
                }
                network_stats.append(stats)
            
            if not network_stats:
                self.logger.warning("No network data for analysis")
                return
            
            stats_df = pd.DataFrame(network_stats)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Total Network Length', 'Electrification Rate', 
                               'High Speed Rail Coverage', 'Network Density'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Total length
            fig.add_trace(
                go.Bar(x=stats_df['country'], y=stats_df['total_length'],
                       name='Total Length (km)', marker_color='lightblue'),
                row=1, col=1
            )
            
            # 2. Electrification rate
            fig.add_trace(
                go.Bar(x=stats_df['country'], y=stats_df['electrified_ratio'] * 100,
                       name='Electrification %', marker_color='lightgreen'),
                row=1, col=2
            )
            
            # 3. High speed rail
            fig.add_trace(
                go.Bar(x=stats_df['country'], y=stats_df['high_speed_ratio'] * 100,
                       name='High Speed %', marker_color='orange'),
                row=2, col=1
            )
            
            # 4. Network density (segments per 1000km)
            fig.add_trace(
                go.Bar(x=stats_df['country'], y=stats_df['segment_count'] / (stats_df['total_length'] / 1000),
                       name='Segments per 1000km', marker_color='purple'),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="Railway Network Analysis by Country",
                showlegend=False,
                height=800
            )
            
            # Save chart
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_file))
            
            self.logger.info(f"Network analysis chart saved to {output_file}")
            
        except ImportError:
            self.logger.warning("Plotly not available, skipping network analysis chart")
        except Exception as e:
            self.logger.error(f"Error creating network analysis chart: {e}")

if __name__ == "__main__":
    # Example usage
    from src.data_processing.data_loader import DataLoader
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    loader = DataLoader()
    
    # Test with Germany
    country = 'germany'
    country_data = loader.load_all_country_data(country)
    
    if country_data:
        visualizer = RouteVisualizer()
        
        # Create sample optimized routes (normally from route optimizer)
        sample_routes = {
            'route_1': {
                'path': ['Berlin', 'Hamburg', 'Bremen'],
                'total_cost': 150.5,
                'segments': [
                    {'distance_km': 289, 'terrain_cost': 100, 'demand_benefit': 20},
                    {'distance_km': 125, 'terrain_cost': 50, 'demand_benefit': 15}
                ]
            }
        }
        
        # Create interactive map
        if 'cities' in country_data:
            map_file = visualizer.create_interactive_map(
                country_data['cities'], 
                sample_routes,
                country_data.get('railway_network', gpd.GeoDataFrame()),
                country
            )
            print(f"Map created: {map_file}")