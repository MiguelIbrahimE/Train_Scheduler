#!/usr/bin/env python3
"""
Direct Railway Pipeline Runner - Uses existing pipeline/src modules
Generates actual railway connections between cities
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add pipeline src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline', 'src'))

# Import all the pipeline modules
from data_loader import DataLoader
from demand_analysis import DemandAnalyzer
from terrain_analysis import TerrainAnalyzer
from cost_analysis import CostAnalyzer
from route_optimizer import RouteOptimizer
from train_selection import TrainSelector
from station_placement import StationOptimizer
from route_mapping import RouteMapper
from visualizer import Visualizer

def generate_html_report(results, output_path="output/railway_network_report.html"):
    """Generate comprehensive HTML report with railway connections"""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚂 BCPC Railway Network Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f0f2f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 0; text-align: center; }}
        .container {{ max-width: 1400px; margin: -30px auto 40px; background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        .subtitle {{ opacity: 0.9; margin-top: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 25px; border-radius: 12px; text-align: center; }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; margin-bottom: 5px; }}
        .routes {{ margin: 30px 0; }}
        .route-card {{ background: #f8f9fa; border: 2px solid #e9ecef; padding: 20px; border-radius: 10px; margin-bottom: 15px; transition: all 0.3s; }}
        .route-card:hover {{ border-color: #667eea; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.1); }}
        .route-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .route-name {{ font-size: 1.3em; font-weight: bold; color: #2c3e50; }}
        .train-type {{ background: #667eea; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9em; }}
        .route-details {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .detail-item {{ display: flex; align-items: center; gap: 10px; }}
        .detail-label {{ font-weight: 600; color: #6c757d; }}
        .cities {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }}
        .city-card {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px; }}
        .network-summary {{ background: #f8f9fa; padding: 30px; border-radius: 10px; margin: 30px 0; }}
        .timestamp {{ text-align: center; color: #6c757d; margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; }}
        .map-placeholder {{ background: #e9ecef; padding: 100px; text-align: center; border-radius: 10px; margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚂 BCPC Railway Network</h1>
        <div class="subtitle">Lebanese Railway Planning System</div>
    </div>
    
    <div class="container">
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(results.get('cities', []))}</div>
                <div>Cities Connected</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(results.get('routes', []))}</div>
                <div>Railway Routes</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{sum(r.get('distance', 0) for r in results.get('routes', [])):.0f} km</div>
                <div>Total Track Length</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${sum(r.get('cost', 0) for r in results.get('routes', []))/1e6:.1f}M</div>
                <div>Total Investment</div>
            </div>
        </div>
        
        <h2>🛤️ Railway Routes</h2>
        <div class="routes">"""
    
    # Add route cards
    for route in results.get('routes', []):
        train_type = route.get('train_type', 'Standard')
        html_content += f"""
            <div class="route-card">
                <div class="route-header">
                    <div class="route-name">{route.get('from', 'Unknown')} ↔️ {route.get('to', 'Unknown')}</div>
                    <div class="train-type">{train_type}</div>
                </div>
                <div class="route-details">
                    <div class="detail-item">
                        <span class="detail-label">📏 Distance:</span>
                        <span>{route.get('distance', 0):.1f} km</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">💰 Cost:</span>
                        <span>${route.get('cost', 0)/1e6:.2f}M</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">👥 Demand:</span>
                        <span>{route.get('demand', 0):,.0f} passengers/year</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">🏔️ Terrain:</span>
                        <span>{route.get('terrain_type', 'Mixed')}</span>
                    </div>
                </div>
            </div>"""
    
    html_content += """
        </div>
        
        <h2>🏙️ Connected Cities</h2>
        <div class="cities">"""
    
    # Add city information
    for city in results.get('cities', []):
        html_content += f"""
            <div class="city-card">
                <h3>{city.get('name', 'Unknown')}</h3>
                <p>👥 Population: {city.get('population', 0):,}</p>
                <p>📍 Location: {city.get('latitude', 0):.3f}°N, {city.get('longitude', 0):.3f}°E</p>
                <p>🚉 Connections: {city.get('connections', 0)}</p>
            </div>"""
    
    html_content += f"""
        </div>
        
        <div class="network-summary">
            <h2>📊 Network Summary</h2>
            <p><strong>Network Type:</strong> {results.get('network_type', 'Hub and Spoke')}</p>
            <p><strong>Total Population Served:</strong> {sum(c.get('population', 0) for c in results.get('cities', [])):,}</p>
            <p><strong>Average Route Distance:</strong> {sum(r.get('distance', 0) for r in results.get('routes', [])) / max(len(results.get('routes', [])), 1):.1f} km</p>
            <p><strong>Primary Hub:</strong> {results.get('primary_hub', 'Beirut')}</p>
        </div>
        
        <div class="map-placeholder">
            <h3>🗺️ Interactive Map</h3>
            <p>Run the visualizer to generate an interactive map of the railway network</p>
        </div>
        
        <div class="timestamp">
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by BCPC Railway Pipeline
        </div>
    </div>
</body>
</html>"""
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path

def run_railway_pipeline():
    """Run the complete railway pipeline using existing modules"""
    
    print("🚂 BCPC Railway Network Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    # Configuration
    config = {
        'csv_path': 'input/lebanon_cities_2024.csv',
        'output_dir': 'output',
        'cache_dir': 'output/cache',
        'verbose': True
    }
    
    results = {
        'cities': [],
        'routes': [],
        'stations': [],
        'config': config,
        'execution_time': 0
    }
    
    try:
        # Step 1: Load data
        print("\n📊 Loading city data...")
        loader = DataLoader(config)
        cities_df = loader.load_cities(config['csv_path'])
        print(f"✅ Loaded {len(cities_df)} cities")
        
        # Convert to list for results
        results['cities'] = cities_df.to_dict('records')
        
        # Step 2: Analyze demand
        print("\n📈 Analyzing inter-city demand...")
        demand_analyzer = DemandAnalyzer()
        demand_matrix = demand_analyzer.calculate_demand_matrix(cities_df)
        city_pairs = demand_analyzer.get_high_demand_pairs(demand_matrix, cities_df, top_n=15)
        print(f"✅ Identified {len(city_pairs)} high-demand routes")
        
        # Step 3: Analyze terrain
        print("\n🏔️ Analyzing terrain...")
        terrain_analyzer = TerrainAnalyzer(cache_dir=config['cache_dir'])
        terrain_data = {}
        for _, city in cities_df.iterrows():
            try:
                terrain = terrain_analyzer.analyze_city_terrain(
                    city['latitude'], 
                    city['longitude'], 
                    city['name']
                )
                if terrain:
                    terrain_data[city['name']] = terrain
            except:
                print(f"⚠️  Terrain analysis failed for {city['name']}")
        
        # Step 4: Calculate costs and optimize routes
        print("\n💰 Calculating costs and optimizing routes...")
        cost_analyzer = CostAnalyzer()
        route_optimizer = RouteOptimizer()
        
        # Create network
        network = route_optimizer.create_network(cities_df)
        
        # Calculate costs for all edges
        for i, city1 in cities_df.iterrows():
            for j, city2 in cities_df.iterrows():
                if i < j:
                    cost = cost_analyzer.calculate_route_cost(
                        city1.to_dict(), 
                        city2.to_dict(),
                        terrain_data.get(city1['name'], {}),
                        terrain_data.get(city2['name'], {})
                    )
                    network.add_edge(city1['name'], city2['name'], weight=cost['total_cost'])
        
        # Find optimal routes (minimum spanning tree)
        optimal_routes = route_optimizer.find_minimum_spanning_tree(network)
        print(f"✅ Optimized {len(optimal_routes)} railway routes")
        
        # Step 5: Select train types
        print("\n🚊 Selecting train types...")
        train_selector = TrainSelector()
        
        for route in optimal_routes:
            # Get route details from cost analyzer
            city1 = cities_df[cities_df['name'] == route['from']].iloc[0]
            city2 = cities_df[cities_df['name'] == route['to']].iloc[0]
            
            cost_details = cost_analyzer.calculate_route_cost(
                city1.to_dict(), 
                city2.to_dict(),
                terrain_data.get(city1['name'], {}),
                terrain_data.get(city2['name'], {})
            )
            
            # Get demand
            demand = demand_matrix.get((route['from'], route['to']), 0) + \
                     demand_matrix.get((route['to'], route['from']), 0)
            
            # Select train type
            train_type = train_selector.select_train_type(
                cost_details['distance'],
                cost_details.get('terrain_difficulty', 'moderate'),
                'high' if demand > 50000 else 'medium'
            )
            
            # Update route info
            route.update({
                'distance': cost_details['distance'],
                'cost': cost_details['total_cost'],
                'demand': demand,
                'train_type': train_type['name'],
                'terrain_type': cost_details.get('terrain_type', 'mixed')
            })
        
        results['routes'] = optimal_routes
        
        # Step 6: Optimize stations
        print("\n🏢 Optimizing station placement...")
        station_optimizer = StationOptimizer()
        stations = station_optimizer.optimize_stations(cities_df, optimal_routes)
        results['stations'] = stations
        print(f"✅ Optimized {len(stations)} station locations")
        
        # Step 7: Create visualizations
        print("\n🎨 Creating visualizations...")
        visualizer = Visualizer(config['output_dir'])
        
        # Create network map
        try:
            map_file = visualizer.create_network_map(cities_df, optimal_routes, stations)
            print(f"✅ Created interactive map: {map_file}")
            results['map_file'] = str(map_file)
        except Exception as e:
            print(f"⚠️  Map creation failed: {e}")
        
        # Calculate execution time
        results['execution_time'] = time.time() - start_time
        
        # Generate HTML report
        print("\n📄 Generating HTML report...")
        html_path = generate_html_report(results)
        print(f"✅ HTML report: {html_path}")
        
        # Save JSON results
        json_path = os.path.join(config['output_dir'], 'railway_network_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✅ JSON results: {json_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 PIPELINE SUMMARY")
        print("="*60)
        print(f"✅ Cities connected: {len(cities_df)}")
        print(f"✅ Railway routes: {len(optimal_routes)}")
        print(f"✅ Total track length: {sum(r['distance'] for r in optimal_routes):.0f} km")
        print(f"✅ Total investment: ${sum(r['cost'] for r in optimal_routes)/1e6:.1f}M")
        print(f"✅ Execution time: {results['execution_time']:.1f} seconds")
        print(f"\n🌐 Open {html_path} to view the results!")
        
        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open(f"file://{Path(html_path).absolute()}")
        except:
            pass
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the pipeline
    results = run_railway_pipeline()
    
    if results:
        print("\n✅ Railway network generation completed successfully!")
    else:
        print("\n❌ Railway network generation failed!")
        sys.exit(1)