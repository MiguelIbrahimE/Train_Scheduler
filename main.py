#!/usr/bin/env python3
"""
Railway Route Optimization - Main Entry Point
"""

import sys
import logging
import argparse
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing.network_extractor import RailwayNetworkExtractor
from data_processing.terrain_analyzer import TerrainAnalyzer
from data_processing.data_loader import DataLoader
from models.optimization.route_optimizer import RouteOptimizer
from visualization.route_visualizer import RouteVisualizer

def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('railway_optimizer.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path="config/countries_config.yaml"):
    """Load configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing configuration: {e}")
        sys.exit(1)

def fetch_data_command(args):
    """Fetch railway data from online sources"""
    print("🚄 Fetching Railway Data from OpenStreetMap...")
    
    extractor = RailwayNetworkExtractor()
    
    countries = args.countries if args.countries else ['germany', 'switzerland']
    
    results = {}
    for country in countries:
        print(f"\n📍 Fetching data for {country}...")
        
        try:
            network, stations = extractor.save_network_data(country)
            results[country] = {
                'network_segments': len(network),
                'stations': len(stations),
                'success': True
            }
            print(f"✅ {country}: {len(network)} segments, {len(stations)} stations")
            
        except Exception as e:
            results[country] = {'success': False, 'error': str(e)}
            print(f"❌ {country}: {e}")
    
    return results

def analyze_terrain_command(args):
    """Analyze terrain for existing networks"""
    print("🏔️ Analyzing Terrain Characteristics...")
    
    analyzer = TerrainAnalyzer()
    
    countries = args.countries if args.countries else ['germany', 'switzerland']
    
    results = {}
    for country in countries:
        print(f"\n📍 Analyzing terrain for {country}...")
        
        try:
            terrain_analysis = analyzer.save_terrain_analysis(country)
            
            if not terrain_analysis.empty:
                results[country] = {
                    'segments_analyzed': len(terrain_analysis),
                    'avg_elevation': terrain_analysis['mean_elevation'].mean(),
                    'avg_gradient': terrain_analysis['mean_gradient'].mean(),
                    'terrain_types': terrain_analysis['terrain_type'].value_counts().to_dict(),
                    'success': True
                }
                print(f"✅ {country}: Analyzed {len(terrain_analysis)} segments")
            else:
                results[country] = {'success': False, 'error': 'No terrain data'}
                print(f"⚠️ {country}: No terrain data available")
                
        except Exception as e:
            results[country] = {'success': False, 'error': str(e)}
            print(f"❌ {country}: {e}")
    
    return results

def optimize_routes_command(args):
    """Optimize routes between cities"""
    print("🎯 Optimizing Routes...")
    
    loader = DataLoader()
    optimizer = RouteOptimizer()
    
    countries = args.countries if args.countries else ['germany', 'switzerland']
    
    all_results = {}
    
    for country in countries:
        print(f"\n📍 Optimizing routes for {country}...")
        
        try:
            # Load data
            country_data = loader.load_all_country_data(country)
            
            if 'cities' not in country_data:
                print(f"⚠️ {country}: No cities data available")
                continue
            
            cities = country_data['cities']
            existing_network = country_data.get('railway_network', None)
            
            # Define route pairs
            if args.routes:
                # Parse route pairs from command line
                route_pairs = []
                for route in args.routes:
                    try:
                        start, end = route.split('-')
                        route_pairs.append((start.strip(), end.strip()))
                    except ValueError:
                        print(f"⚠️ Invalid route format: {route}")
            else:
                # Use top cities
                cities_sorted = cities.sort_values('population', ascending=False)
                top_cities = cities_sorted.head(6)['city'].tolist()
                
                route_pairs = []
                for i in range(len(top_cities)):
                    for j in range(i + 1, min(i + 4, len(top_cities))):
                        route_pairs.append((top_cities[i], top_cities[j]))
            
            print(f"   Optimizing {len(route_pairs)} route pairs...")
            
            # Optimize routes
            optimized_routes = optimizer.optimize_multiple_routes(
                cities, route_pairs, existing_network
            )
            
            if optimized_routes:
                all_results[country] = optimized_routes
                
                # Generate and save report
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
                if not report.empty:
                    print(f"   Average cost: {report['total_cost'].mean():.2f}")
                    print(f"   Average distance: {report['total_distance_km'].mean():.1f} km")
                    print(f"   Average efficiency: {report['route_efficiency'].mean():.1%}")
            else:
                print(f"⚠️ {country}: No routes optimized")
                
        except Exception as e:
            print(f"❌ {country}: {e}")
            continue
    
    return all_results

def visualize_command(args):
    """Create visualizations"""
    print("🗺️ Creating Visualizations...")
    
    loader = DataLoader()
    visualizer = RouteVisualizer()
    
    countries = args.countries if args.countries else ['germany', 'switzerland']
    
    for country in countries:
        print(f"\n📍 Creating visualization for {country}...")
        
        try:
            # Load data
            country_data = loader.load_all_country_data(country)
            
            if 'cities' not in country_data:
                print(f"⚠️ {country}: No cities data available")
                continue
            
            # Load optimized routes (if available)
            routes_file = Path(f"data/output/routes/{country}_optimized_routes.geojson")
            if routes_file.exists():
                # Load routes from GeoJSON and convert to expected format
                import json
                with open(routes_file, 'r') as f:
                    routes_geojson = json.load(f)
                
                # Convert to expected format (simplified)
                routes = {}
                for i, feature in enumerate(routes_geojson['features']):
                    route_id = feature['properties'].get('route_id', f'route_{i}')
                    routes[route_id] = {
                        'path': feature['properties'].get('path', []),
                        'total_cost': feature['properties'].get('total_cost', 0),
                        'segments': []  # Simplified
                    }
            else:
                routes = {}
                print(f"⚠️ {country}: No optimized routes found")
            
            # Create interactive map
            map_file = visualizer.create_interactive_map(
                country_data['cities'], 
                routes,
                country_data.get('railway_network', None),
                country
            )
            
            print(f"✅ {country}: Map saved to {map_file}")
            
        except Exception as e:
            print(f"❌ {country}: {e}")
            continue

def full_pipeline_command(args):
    """Run the complete pipeline"""
    print("🚄 Running Complete Railway Route Optimization Pipeline")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("\n1️⃣ Fetching railway data...")
    fetch_data_command(args)
    
    # Step 2: Analyze terrain
    print("\n2️⃣ Analyzing terrain...")
    analyze_terrain_command(args)
    
    # Step 3: Optimize routes
    print("\n3️⃣ Optimizing routes...")
    optimize_routes_command(args)
    
    # Step 4: Create visualizations
    print("\n4️⃣ Creating visualizations...")
    visualize_command(args)
    
    print("\n🎉 Pipeline completed successfully!")
    print("📁 Results saved to data/output/")

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Railway Route Optimization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py fetch --countries germany switzerland
  python main.py optimize --routes "Berlin-Munich" "Hamburg-Frankfurt"
  python main.py visualize --countries germany
  python main.py full-pipeline --countries germany switzerland
        """
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--countries',
        nargs='+',
        help='Countries to process (default: germany switzerland)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fetch data command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch railway data from online sources')
    fetch_parser.set_defaults(func=fetch_data_command)
    
    # Analyze terrain command
    terrain_parser = subparsers.add_parser('terrain', help='Analyze terrain characteristics')
    terrain_parser.set_defaults(func=analyze_terrain_command)
    
    # Optimize routes command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize routes between cities')
    optimize_parser.add_argument(
        '--routes',
        nargs='+',
        help='Route pairs in format "City1-City2" "City3-City4"'
    )
    optimize_parser.set_defaults(func=optimize_routes_command)
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Create visualizations')
    visualize_parser.set_defaults(func=visualize_command)
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('full-pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument(
        '--routes',
        nargs='+',
        help='Route pairs in format "City1-City2" "City3-City4"'
    )
    pipeline_parser.set_defaults(func=full_pipeline_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config()
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\n⏹️ Operation interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Operation failed: {e}")
            logging.error(f"Operation failed: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()