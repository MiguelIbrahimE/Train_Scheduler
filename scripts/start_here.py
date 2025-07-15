#!/usr/bin/env python3
"""
Start Here - Basic Railway Data Fetching and Route Optimization
This script demonstrates the complete pipeline from data fetching to route optimization.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import geopandas as gpd
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.network_extractor import RailwayNetworkExtractor
from data_processing.terrain_analyzer import TerrainAnalyzer
from data_processing.data_loader import DataLoader
from models.optimization.route_optimizer import RouteOptimizer
from visualization.route_visualizer import RouteVisualizer

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('railway_optimization.log'),
            logging.StreamHandler()
        ]
    )

def fetch_railway_data(countries: list = None):
    """Step 1: Fetch railway data from OpenStreetMap"""
    
    if countries is None:
        countries = ['germany', 'switzerland']  # Start with these two
    
    print("🚄 Step 1: Fetching Railway Data from OpenStreetMap...")
    print("=" * 50)
    
    extractor = RailwayNetworkExtractor()
    
    for country in countries:
        print(f"\n📍 Fetching data for {country}...")
        
        try:
            # Extract and save network data
            network, stations = extractor.save_network_data(country)
            
            print(f"✅ {country}: {len(network)} railway segments, {len(stations)} stations")
            
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error fetching {country}: {e}")
            continue
    
    print("\n✅ Railway data fetching complete!")

def analyze_terrain(countries: list = None):
    """Step 2: Analyze terrain characteristics"""
    
    if countries is None:
        countries = ['germany', 'switzerland']
    
    print("\n🏔️ Step 2: Analyzing Terrain Characteristics...")
    print("=" * 50)
    
    terrain_analyzer = TerrainAnalyzer()
    
    for country in countries:
        print(f"\n📍 Analyzing terrain for {country}...")
        
        try:
            # Analyze terrain for existing network
            terrain_analysis = terrain_analyzer.save_terrain_analysis(country)
            
            if not terrain_analysis.empty:
                print(f"✅ {country}: Analyzed {len(terrain_analysis)} route segments")
                
                # Print some basic stats
                avg_elevation = terrain_analysis['mean_elevation'].mean()
                avg_gradient = terrain_analysis['mean_gradient'].mean()
                terrain_types = terrain_analysis['terrain_type'].value_counts()
                
                print(f"   Average elevation: {avg_elevation:.0f}m")
                print(f"   Average gradient: {avg_gradient:.1%}")
                print(f"   Terrain types: {dict(terrain_types)}")
            else:
                print(f"⚠️ {country}: No terrain analysis data")
            
        except Exception as e:
            print(f"❌ Error analyzing terrain for {country}: {e}")
            continue
    
    print("\n✅ Terrain analysis complete!")

def load_and_process_data(countries: list = None):
    """Step 3: Load and process all data"""
    
    if countries is None:
        countries = ['germany', 'switzerland']
    
    print("\n📊 Step 3: Loading and Processing Data...")
    print("=" * 50)
    
    loader = DataLoader()
    
    all_data = {}
    
    for country in countries:
        print(f"\n📍 Loading data for {country}...")
        
        try:
            country_data = loader.load_all_country_data(country)
            
            if country_data:
                all_data[country] = country_data
                
                # Print data summary
                for dataset_name, dataset in country_data.items():
                    print(f"   {dataset_name}: {len(dataset)} records")
                
                print(f"✅ {country}: {len(country_data)} datasets loaded")
            else:
                print(f"⚠️ {country}: No data loaded")
                
        except Exception as e:
            print(f"❌ Error loading data for {country}: {e}")
            continue
    
    print("\n✅ Data loading complete!")
    return all_data

def optimize_routes(all_data: dict):
    """Step 4: Optimize routes between cities"""
    
    print("\n🎯 Step 4: Optimizing Routes...")
    print("=" * 50)
    
    optimizer = RouteOptimizer()
    
    all_results = {}
    
    for country, country_data in all_data.items():
        if 'cities' not in country_data:
            print(f"⚠️ {country}: No cities data available")
            continue
        
        print(f"\n📍 Optimizing routes for {country}...")
        
        cities = country_data['cities']
        existing_network = country_data.get('railway_network', gpd.GeoDataFrame())
        
        # Define route pairs based on largest cities
        cities_sorted = cities.sort_values('population', ascending=False)
        top_cities = cities_sorted.head(6)['city'].tolist()
        
        # Create route pairs between top cities
        route_pairs = []
        for i in range(len(top_cities)):
            for j in range(i + 1, min(i + 4, len(top_cities))):  # Connect to next 3 cities
                route_pairs.append((top_cities[i], top_cities[j]))
        
        print(f"   Optimizing {len(route_pairs)} route pairs...")
        
        try:
            # Optimize routes
            optimized_routes = optimizer.optimize_multiple_routes(
                cities, route_pairs, existing_network
            )
            
            if optimized_routes:
                all_results[country] = optimized_routes
                
                # Generate report
                report = optimizer.generate_optimization_report(optimized_routes, cities, country)
                
                # Save results
                output_dir = Path("data/output")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Export to GeoJSON
                optimizer.export_routes_to_geojson(
                    optimized_routes, cities, 
                    f"data/output/routes/{country}_optimized_routes.geojson"
                )
                
                # Save report
                report.to_csv(f"data/output/reports/{country}_optimization_report.csv", index=False)
                
                print(f"✅ {country}: Optimized {len(optimized_routes)} routes")
                
                # Print summary
                avg_cost = report['total_cost'].mean()
                avg_distance = report['total_distance_km'].mean()
                avg_efficiency = report['route_efficiency'].mean()
                
                print(f"   Average cost: {avg_cost:.2f}")
                print(f"   Average distance: {avg_distance:.1f} km")
                print(f"   Average efficiency: {avg_efficiency:.1%}")
                
            else:
                print(f"⚠️ {country}: No routes optimized")
                
        except Exception as e:
            print(f"❌ Error optimizing routes for {country}: {e}")
            continue
    
    print("\n✅ Route optimization complete!")
    return all_results

def create_visualizations(all_data: dict, all_results: dict):
    """Step 5: Create visualizations"""
    
    print("\n🗺️ Step 5: Creating Visualizations...")
    print("=" * 50)
    
    try:
        from visualization.route_visualizer import RouteVisualizer
        visualizer = RouteVisualizer()
        
        for country in all_data.keys():
            if country not in all_results:
                continue
                
            print(f"\n📍 Creating visualization for {country}...")
            
            cities = all_data[country]['cities']
            routes = all_results[country]
            existing_network = all_data[country].get('railway_network', gpd.GeoDataFrame())
            
            # Create map
            map_file = visualizer.create_interactive_map(
                cities, routes, existing_network, country
            )
            
            print(f"✅ {country}: Map saved to {map_file}")
            
    except ImportError:
        print("⚠️ Visualization module not ready yet - skipping visualizations")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

def main():
    """Main execution function"""
    
    print("🚄 Railway Route Optimization Pipeline")
    print("=" * 50)
    print("This script will:")
    print("1. 📡 Fetch railway data from OpenStreetMap")
    print("2. 🏔️ Analyze terrain characteristics")
    print("3. 📊 Load and process all data")
    print("4. 🎯 Optimize routes between cities")
    print("5. 🗺️ Create visualizations")
    print("=" * 50)
    
    # Setup
    setup_logging()
    
    # Countries to process (start with best railway systems)
    countries = ['germany', 'switzerland']
    
    try:
        # Step 1: Fetch railway data
        fetch_railway_data(countries)
        
        # Step 2: Analyze terrain
        analyze_terrain(countries)
        
        # Step 3: Load and process data
        all_data = load_and_process_data(countries)
        
        # Step 4: Optimize routes
        if all_data:
            all_results = optimize_routes(all_data)
            
            # Step 5: Create visualizations
            if all_results:
                create_visualizations(all_data, all_results)
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"📁 Results saved to: data/output/")
        print(f"📊 Reports: data/output/reports/")
        print(f"🗺️ Maps: data/output/routes/")
        print(f"📝 Logs: railway_optimization.log")
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()