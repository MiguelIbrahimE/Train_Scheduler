#!/usr/bin/env python3
"""
BCPC Pipeline Initiator
======================

Complete pipeline that follows the BCPC workflow diagram:
1. Read CSV → 2. Understand population → 3. Understand demand → 4. Understand costs 
→ 5. Understand terrain → 6. Map routes → 7. Optimize → 8. Visualize

Usage:
    python pipeline_initiator.py --csv input/lebanon_cities_2024.csv
    python pipeline_initiator.py --csv input/lebanon_cities_2024.csv --output dashboard/
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, time 

# Setup path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    # Import the existing data_loader
    from data_loader import DataLoader, CityData
    
    # Import new pipeline modules
    from terrain_analysis import TerrainAnalyzer, DEMSource, TerrainComplexity
    from station_placement import (
        StationPlacementOptimizer, PopulationCenter, EmploymentCenter,
        create_example_city_data
    )
    from cost_analysis import (
        CostAnalyzer, NetworkDesign, TrainType, TrackGauge, 
        estimate_cost
    )
    from visualizer import (
        create_complete_dashboard, BCPCVisualizer,
        create_visualization_from_scenario
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all pipeline modules are in the pipeline/src/ directory")
    print("Current working directory:", Path.cwd())
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BCPCPipelineInitiator:
    """
    Complete BCPC analysis pipeline following the workflow diagram
    """
    
    def __init__(self, output_dir: str = "output", max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        # Step 1: Initialize data loader
        self.data_loader = DataLoader(cache_dir=str(self.output_dir / "cache"))
        
        # Initialize other pipeline components
        self.terrain_analyzer = TerrainAnalyzer(
            cache_dir="data/_cache/terrain",
            preferred_resolution=250.0  # Use 250m resolution for OpenElevation
        )
        
        self.station_optimizer = StationPlacementOptimizer()
        self.cost_analyzer = CostAnalyzer()
        self.visualizer = BCPCVisualizer()
        
        # Pipeline results storage
        self.pipeline_results = {}
    
    def run_complete_pipeline(self, csv_path: str) -> Dict[str, Any]:
        """
        Run the complete BCPC pipeline following the workflow diagram
        
        Args:
            csv_path: Path to the input CSV file
            
        Returns:
            Complete pipeline results
        """
        logger.info("=" * 60)
        logger.info("STARTING BCPC PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Read CSV
            logger.info("STEP 1: Reading CSV data...")
            cities_data = self._step1_read_csv(csv_path)
            
            # Step 2: Understand population (and demand sources)
            logger.info("STEP 2: Understanding population and demand sources...")
            enriched_cities = self._step2_understand_population(cities_data)
            
            # Step 3: Understand demand
            logger.info("STEP 3: Understanding demand patterns...")
            demand_analysis = self._step3_understand_demand(enriched_cities)
            
            # Step 4: Understand costs (Rail, Train Types, Railway labor costs, electricity)
            logger.info("STEP 4: Understanding costs...")
            cost_framework = self._step4_understand_costs(enriched_cities)
            
            # Step 5: Understand terrain using open source
            logger.info("STEP 5: Understanding terrain...")
            terrain_results = self._step5_understand_terrain(enriched_cities)
            
            # Step 6: Map some sample routes with tunnels and elevated sections
            logger.info("STEP 6: Mapping sample routes...")
            route_mapping = self._step6_map_routes(enriched_cities, terrain_results)
            
            # Step 7: Optimize Route with Range Dates
            logger.info("STEP 7: Optimizing routes...")
            optimization_results = self._step7_optimize_routes(
                enriched_cities, terrain_results, route_mapping, demand_analysis, cost_framework
            )
            
            # Step 8: Visualize map using HTML
            logger.info("STEP 8: Creating visualizations...")
            visualization_results = self._step8_visualize_map(optimization_results)
            
            # Compile final results
            final_results = {
                'input_data': {
                    'csv_path': csv_path,
                    'cities_count': len(cities_data),
                    'data_summary': self.data_loader.get_data_summary(cities_data)
                },
                'cities_data': enriched_cities,
                'demand_analysis': demand_analysis,
                'cost_framework': cost_framework,
                'terrain_results': terrain_results,
                'route_mapping': route_mapping,
                'optimization_results': optimization_results,
                'visualization_results': visualization_results
            }
            
            # Save pipeline results
            self._save_pipeline_results(final_results)
            
            logger.info("=" * 60)
            logger.info("BCPC PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _step1_read_csv(self, csv_path: str) -> List[CityData]:
        """Step 1: Read CSV with data loader"""
        
        logger.info(f"Loading and validating CSV data from {csv_path}")
        
        # Use the existing data loader
        cities_data = self.data_loader.load_csv(csv_path)
        valid_cities = self.data_loader.validate_city_data(cities_data)
        
        if not valid_cities:
            raise ValueError("No valid cities found in CSV data")
        
        logger.info(f"Successfully loaded {len(valid_cities)} valid cities")
        
        # Log data summary
        summary = self.data_loader.get_data_summary(valid_cities)
        logger.info(f"Data summary: {summary}")
        
        return valid_cities
    
    def _step2_understand_population(self, cities_data: List[CityData]) -> Dict[str, Any]:
        """Step 2: Understand population and create enhanced city data"""
        
        logger.info("Analyzing population distribution and creating city boundaries...")
        
        enriched_cities = {}
        
        for city in cities_data:
            logger.info(f"Processing city: {city.name}")
            
            # Create city boundary (buffer around center point)
            center_point = Point(city.longitude, city.latitude)
            city_boundary = center_point.buffer(0.05)  # ~5km radius
            
            # Create population centers based on city data
            population_centers = self._create_population_centers(city, center_point)
            
            # Create employment centers
            employment_centers = self._create_employment_centers(city, center_point)
            
            enriched_cities[city.name] = {
                'original_data': city,
                'center_point': center_point,
                'boundary': city_boundary,
                'population_centers': population_centers,
                'employment_centers': employment_centers,
                'total_population': city.population,
                'budget_eur': city.budget
            }
        
        logger.info(f"Enriched {len(enriched_cities)} cities with population analysis")
        return enriched_cities
    
    def _step3_understand_demand(self, enriched_cities: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Understand demand patterns between cities"""
        
        logger.info("Analyzing travel demand patterns...")
        
        demand_analysis = {
            'intercity_demand': {},
            'total_corridor_demand': 0,
            'peak_demand_routes': [],
            'demand_methodology': 'gravity_model'
        }
        
        city_names = list(enriched_cities.keys())
        
        # Calculate intercity demand using gravity model
        for i, city_a in enumerate(city_names):
            for j, city_b in enumerate(city_names[i+1:], i+1):
                
                city_a_data = enriched_cities[city_a]
                city_b_data = enriched_cities[city_b]
                
                # Calculate distance between cities
                distance_km = city_a_data['center_point'].distance(
                    city_b_data['center_point']
                ) * 111  # Convert degrees to km (rough)
                
                if distance_km > 0:
                    # Gravity model: demand proportional to population product, inversely to distance squared
                    pop_a = city_a_data['total_population']
                    pop_b = city_b_data['total_population']
                    
                    # Simplified gravity model
                    gravity_constant = 0.00001  # Calibration factor
                    daily_demand = (gravity_constant * pop_a * pop_b) / (distance_km ** 1.5)
                    
                    # Apply tourism factor
                    tourism_factor = (
                        city_a_data['original_data'].tourism_index + 
                        city_b_data['original_data'].tourism_index
                    ) / 2
                    
                    adjusted_demand = daily_demand * (1 + tourism_factor)
                    
                    route_key = f"{city_a}-{city_b}"
                    demand_analysis['intercity_demand'][route_key] = {
                        'daily_passengers': int(adjusted_demand),
                        'distance_km': distance_km,
                        'population_a': pop_a,
                        'population_b': pop_b,
                        'tourism_factor': tourism_factor
                    }
                    
                    demand_analysis['total_corridor_demand'] += adjusted_demand
        
        # Identify peak demand routes
        sorted_routes = sorted(
            demand_analysis['intercity_demand'].items(),
            key=lambda x: x[1]['daily_passengers'],
            reverse=True
        )
        
        demand_analysis['peak_demand_routes'] = sorted_routes[:5]  # Top 5 routes
        
        logger.info(f"Analyzed demand for {len(demand_analysis['intercity_demand'])} city pairs")
        logger.info(f"Total corridor demand: {demand_analysis['total_corridor_demand']:,.0f} daily passengers")
        
        return demand_analysis
    
    def _step4_understand_costs(self, enriched_cities: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Understand costs (Rail, Train Types, Railway labor costs, electricity)"""
        
        logger.info("Analyzing cost framework...")
        
        # Initialize cost analyzer to get baseline costs
        cost_framework = {
            'base_costs': {
                'track_construction_flat_eur_per_km': 2_500_000,
                'track_construction_rolling_eur_per_km': 4_000_000,
                'track_construction_mountainous_eur_per_km': 8_000_000,
                'electrification_eur_per_km': 750_000,
                'signaling_eur_per_km': 500_000
            },
            'train_types': {
                TrainType.DIESEL: {
                    'cost_per_trainset_eur': 3_500_000,
                    'energy_cost_per_km': 2.5,  # EUR per km
                    'maintenance_factor': 0.08
                },
                TrainType.ELECTRIC_EMU: {
                    'cost_per_trainset_eur': 4_200_000,
                    'energy_cost_per_km': 1.8,  # EUR per km
                    'maintenance_factor': 0.06
                },
                TrainType.ELECTRIC_LOCOMOTIVE: {
                    'cost_per_trainset_eur': 5_000_000,
                    'energy_cost_per_km': 2.0,  # EUR per km
                    'maintenance_factor': 0.07
                }
            },
            'operational_costs': {
                'staff_cost_eur_per_year': 65_000,
                'track_maintenance_eur_per_km_per_year': 50_000,
                'energy_cost_electric_eur_per_kwh': 0.12,
                'energy_cost_diesel_eur_per_liter': 1.2
            },
            'regional_adjustments': {}
        }
        
        # Apply regional cost adjustments based on city data
        for city_name, city_data in enriched_cities.items():
            country = city_data['original_data'].country
            
            # Simple regional cost adjustments (would be more sophisticated in reality)
            if country.lower() in ['lebanon', 'jordan', 'syria']:
                adjustment_factor = 0.7  # 30% lower costs in Middle East
            elif country.lower() in ['germany', 'france', 'switzerland']:
                adjustment_factor = 1.3  # 30% higher costs in Western Europe
            else:
                adjustment_factor = 1.0  # Base costs
            
            cost_framework['regional_adjustments'][country] = adjustment_factor
        
        logger.info("Cost framework analysis completed")
        return cost_framework
    
    def _step5_understand_terrain(self, enriched_cities: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Understand terrain using parallel processing"""
        
        logger.info(f"Analyzing terrain for {len(enriched_cities)} cities using {self.max_workers} workers...")
        
        terrain_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all terrain analysis tasks
            future_to_city = {
                executor.submit(self._analyze_city_terrain, city_name, city_data): city_name
                for city_name, city_data in enriched_cities.items()
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_city):
                city_name = future_to_city[future]
                try:
                    result_city, result_data = future.result()
                    terrain_results[result_city] = result_data
                    completed += 1
                    logger.info(f"Progress: {completed}/{len(enriched_cities)} cities analyzed")
                except Exception as e:
                    logger.error(f"Failed to get terrain result for {city_name}: {e}")
                    # Add fallback data
                    terrain_results[city_name] = {
                        'terrain_analysis': None,
                        'route_line': None,
                        'complexity': TerrainComplexity.FLAT,
                        'cost_multiplier': 1.0,
                        'feasibility': 0.8
                    }
        
        logger.info(f"Terrain analysis completed for {len(terrain_results)} cities")
        return terrain_results
        # single-city terrain task for the thread-pool
    def _analyze_city_terrain(self, city_name: str, city_data: Dict[str, Any]):
        center = city_data['center_point']
        city_route = LineString([
            (center.x - 0.02, center.y - 0.02),
            (center.x,         center.y),
            (center.x + 0.02,  center.y + 0.02)
        ])

        try:
            ta = self.terrain_analyzer.analyze_route_terrain(
                route_line=city_route,
                buffer_km=3.0
            )
            result = {
                'terrain_analysis': ta,
                'route_line': city_route,
                'complexity': ta.overall_complexity,
                'cost_multiplier': ta.cost_multiplier,
                'feasibility': ta.construction_feasibility
            }
        except Exception as e:
            logger.warning("Terrain analysis failed for %s: %s", city_name, e)
            result = {
                'terrain_analysis': None,
                'route_line': city_route,
                'complexity': TerrainComplexity.FLAT,
                'cost_multiplier': 1.0,
                'feasibility': 0.8
            }

        return city_name, result

    def _step6_map_routes(self,
                          enriched_cities: Dict[str, Any],
                          terrain_results: Dict[str, Any]) -> Dict[str, Any]:

        logger.info("Mapping sample routes between cities in parallel…")
        route_mapping = {'corridor_routes': {}, 'station_networks': {}}

        city_names = list(enriched_cities.keys())
        if len(city_names) < 2:
            return route_mapping

        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            for i in range(len(city_names) - 1):
                a, b = city_names[i], city_names[i + 1]
                tasks.append(exe.submit(self._map_single_route, a, b, enriched_cities))

            done = 0
            for fut in as_completed(tasks):
                route_key, data, sta_a, sta_b = fut.result()
                route_mapping['corridor_routes'][route_key] = data
                route_mapping['station_networks'][sta_a[0]] = sta_a[1]
                route_mapping['station_networks'][sta_b[0]] = sta_b[1]
                done += 1
                logger.info(f"Route progress: {done}/{len(tasks)}")

        logger.info(f"Mapped {len(route_mapping['corridor_routes'])} corridor routes")
        return route_mapping

    
    def _step7_optimize_routes(self, enriched_cities: Dict[str, Any],
                              terrain_results: Dict[str, Any],
                              route_mapping: Dict[str, Any],
                              demand_analysis: Dict[str, Any],
                              cost_framework: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Optimize Route with Range Dates"""
        
        logger.info("Optimizing routes based on demand, costs, and terrain...")
        
        optimization_results = {
            'optimized_routes': {},
            'cost_estimates': {},
            'recommended_train_types': {},
            'total_network_cost': 0
        }
        
        # Optimize each corridor route
        for route_key, route_data in route_mapping['corridor_routes'].items():
            logger.info(f"Optimizing route: {route_key}")
            
            # Get demand for this route
            route_demand = demand_analysis['intercity_demand'].get(route_key, {})
            daily_passengers = route_demand.get('daily_passengers', 1000)
            
            # Determine optimal train type based on demand and distance
            distance_km = route_data['distance_km']
            train_type = self._select_optimal_train_type(daily_passengers, distance_km)
            
            # Create network design
            network_design = NetworkDesign(
                route_length_km=distance_km,
                track_gauge=TrackGauge.STANDARD,
                train_type=train_type,
                number_of_trainsets=max(2, int(daily_passengers / 2000)),  # Rough sizing
                electrification_required=(train_type != TrainType.DIESEL),
                number_of_stations=4,  # Default for intercity route
                major_stations=2,
                terrain_complexity=TerrainComplexity.ROLLING,  # Default
                daily_passengers_per_direction=daily_passengers,
                operating_speed_kmh=120
            )
            
            # Apply terrain complexity if available
            if route_data['terrain_analysis']:
                network_design.terrain_complexity = route_data['terrain_analysis'].overall_complexity
            
            # Calculate costs
            cost_summary = self.cost_analyzer.analyze_full_cost(
                network_design=network_design,
                budget_constraint=None
            )
            
            optimization_results['optimized_routes'][route_key] = {
                'network_design': network_design,
                'route_line': route_data['route_line'],
                'terrain_analysis': route_data['terrain_analysis'],
                'daily_passengers': daily_passengers
            }
            
            optimization_results['cost_estimates'][route_key] = cost_summary
            optimization_results['recommended_train_types'][route_key] = train_type
            optimization_results['total_network_cost'] += cost_summary.total_capex
        
        logger.info(f"Route optimization completed. Total network cost: €{optimization_results['total_network_cost']:,.0f}")
        return optimization_results
    
    def _step8_visualize_map(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 8: Visualize map using HTML"""
        
        logger.info("Creating interactive visualizations...")
        
        # Prepare data for visualization
        scenario_results = {}
        
        for route_key, route_opt in optimization_results['optimized_routes'].items():
            scenario_results[route_key] = {
                'route_line': route_opt['route_line'],
                'terrain_analysis': route_opt['terrain_analysis'],
                'station_network': None,  # Would be populated from station optimization
                'cost_summary': optimization_results['cost_estimates'][route_key],
                'network_design': route_opt['network_design']
            }
        
        # Create visualizations
        output_dir = self.output_dir / "visualizations"
        output_dir.mkdir(exist_ok=True)
        
        # Main visualization
        main_viz_path = output_dir / "railway_network.html"
        create_visualization_from_scenario(scenario_results, str(main_viz_path))
        
        # Complete dashboard
        create_complete_dashboard(scenario_results, str(output_dir))
        
        visualization_results = {
            'main_visualization': str(main_viz_path),
            'dashboard_directory': str(output_dir),
            'files_created': [
                str(main_viz_path),
                str(output_dir / "index.html"),
                str(output_dir / "main_map.html")
            ]
        }
        
        logger.info(f"Visualizations created in {output_dir}")
        return visualization_results
    
    # Helper methods
    
    def _create_population_centers(self, city: CityData, center_point: Point) -> List[PopulationCenter]:
        """Create population centers for a city"""
        
        population_centers = []
        
        # Main city center (40% of population)
        population_centers.append(PopulationCenter(
            location=center_point,
            population=int(city.population * 0.4),
            density_per_km2=5000,
            area_km2=city.population / 5000 * 0.4,
            center_type='mixed'
        ))
        
        # Suburban areas (60% distributed around)
        for i, (dx, dy) in enumerate([(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]):
            suburban_point = Point(center_point.x + dx, center_point.y + dy)
            population_centers.append(PopulationCenter(
                location=suburban_point,
                population=int(city.population * 0.15),
                density_per_km2=2500,
                area_km2=city.population / 2500 * 0.15,
                center_type='residential'
            ))
        
        return population_centers
    
    def _create_employment_centers(self, city: CityData, center_point: Point) -> List[EmploymentCenter]:
        """Create employment centers for a city"""
        
        employment_centers = []
        
        # CBD (70% of jobs)
        employment_centers.append(EmploymentCenter(
            location=center_point,
            job_count=int(city.population * 0.4 * 0.7),  # 40% employment rate, 70% in CBD
            business_type='cbd',
            area_km2=5,
            daily_workers=int(city.population * 0.4 * 0.7 * 1.2)
        ))
        
        # Industrial zone (30% of jobs)
        industrial_point = Point(center_point.x + 0.03, center_point.y - 0.02)
        employment_centers.append(EmploymentCenter(
            location=industrial_point,
            job_count=int(city.population * 0.4 * 0.3),
            business_type='industrial',
            area_km2=10,
            daily_workers=int(city.population * 0.4 * 0.3 * 1.1)
        ))
        
        return employment_centers
    
    def _optimize_city_stations(self, city_name: str, city_data: Dict[str, Any], 
                               route_line: LineString):
        """Optimize station placement for a city"""
        
        try:
            return self.station_optimizer.optimize_city_stations(
                city_boundary=city_data['boundary'],
                population_centers=city_data['population_centers'],
                employment_centers=city_data['employment_centers'],
                route_line=route_line,
                city_name=city_name,
                max_stations=3
            )
        except Exception as e:
            logger.warning(f"Station optimization failed for {city_name}: {e}")
            return None
    
        # ─────────────────────────── parallel route helper ──────────────────────────
    def _map_single_route(self,
                          city_a: str,
                          city_b: str,
                          enriched: Dict[str, Any]) -> Tuple[str, Dict, Tuple[str, Any], Tuple[str, Any]]:
        point_a = enriched[city_a]['center_point']
        point_b = enriched[city_b]['center_point']
        route_line = LineString([point_a.coords[0], point_b.coords[0]])

        try:
            terrain = self.terrain_analyzer.analyze_route_terrain(route_line, buffer_km=2.0)
        except Exception as exc:
            logger.warning(f"Terrain analysis failed for route {city_a}-{city_b}: {exc}")
            terrain = None

        stations_a = self._optimize_city_stations(city_a, enriched[city_a], route_line)
        stations_b = self._optimize_city_stations(city_b, enriched[city_b], route_line)

        route_key = f"{city_a}-{city_b}"
        data = {
            'route_line':   route_line,
            'terrain_analysis': terrain,
            'distance_km':  route_line.length * 111,
            'city_a':       city_a,
            'city_b':       city_b
        }
        return route_key, data, (city_a, stations_a), (city_b, stations_b)

    def _select_optimal_train_type(self, daily_passengers: int, distance_km: float) -> TrainType:
        """Select optimal train type based on demand and distance"""
        
        if daily_passengers > 15000:
            return TrainType.ELECTRIC_EMU  # High capacity electric
        elif distance_km > 100:
            return TrainType.ELECTRIC_LOCOMOTIVE  # Long distance
        elif daily_passengers < 5000:
            return TrainType.DIESEL  # Low demand, flexible
        else:
            return TrainType.ELECTRIC_EMU  # Default modern option
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file"""
        
        output_file = self.output_dir / "pipeline_results.json"
        
        # Convert complex objects to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Pipeline results saved to {output_file}")
    
    def _make_serializable(self, obj):
        """Convert complex objects to JSON-serializable format"""
        
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'wkt'):  # Shapely geometries
            return obj.wkt
        else:
            return obj


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='BCPC Pipeline Initiator')
    parser.add_argument('--csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize and run pipeline
        pipeline = BCPCPipelineInitiator(output_dir=args.output)
        results = pipeline.run_complete_pipeline(args.csv)
        
        print("\n" + "=" * 60)
        print("BCPC PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {args.output}")
        print(f"Visualizations available at: {args.output}/visualizations/index.html")
        print(f"Total cities processed: {results['input_data']['cities_count']}")
        print(f"Total routes analyzed: {len(results['route_mapping']['corridor_routes'])}")
        print(f"Total network cost: €{results['optimization_results']['total_network_cost']:,.0f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()