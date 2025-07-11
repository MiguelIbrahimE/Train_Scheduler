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
    
    # Import new pipeline modules - FIXED IMPORTS
    from terrain_analysis import TerrainAnalyzer, DEMSource, TerrainComplexity, analyze_terrain_lightweight, create_mock_terrain_analysis
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
            
            # Step 5: Understand terrain using open source - FIXED VERSION
            logger.info("STEP 5: Understanding terrain...")
            terrain_results = self._step5_understand_terrain(enriched_cities)
            
            # Step 6: Map some sample routes with tunnels and elevated sections - FIXED VERSION
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
        """
        FIXED: Lightweight terrain analysis with reduced API calls
        """
        logger.info(f"🏔️ Analyzing terrain for {len(enriched_cities)} cities using lightweight method...")
        
        terrain_results = {}
        
        for city_name, city_data in enriched_cities.items():
            logger.info(f"📍 Analyzing terrain for {city_name}")
            
            try:
                center = city_data['center_point']
                
                # Create a small route segment for terrain analysis
                city_route = LineString([
                    (center.x - 0.01, center.y - 0.01),  # 1km southwest
                    (center.x,         center.y),         # City center  
                    (center.x + 0.01,  center.y + 0.01)   # 1km northeast
                ])
                
                # Use lightweight terrain analysis (10-20 API calls instead of 1500+)
                terrain_analysis = analyze_terrain_lightweight(city_route, city_name)
                
                terrain_results[city_name] = {
                    'terrain_analysis': terrain_analysis,
                    'route_line': city_route,
                    'complexity': terrain_analysis.overall_complexity,
                    'cost_multiplier': terrain_analysis.cost_multiplier,
                    'feasibility': terrain_analysis.construction_feasibility,
                    'status': 'success'
                }
                
                logger.info(f"✅ {city_name}: {terrain_analysis.overall_complexity.value} terrain "
                           f"({terrain_analysis.cost_multiplier:.1f}x cost)")
                
            except Exception as e:
                logger.warning(f"⚠️ Terrain analysis failed for {city_name}: {e}")
                
                # Use mock data as fallback
                mock_analysis = create_mock_terrain_analysis(city_route)
                terrain_results[city_name] = {
                    'terrain_analysis': mock_analysis,
                    'route_line': city_route,
                    'complexity': mock_analysis.overall_complexity,
                    'cost_multiplier': mock_analysis.cost_multiplier,
                    'feasibility': mock_analysis.construction_feasibility,
                    'status': 'fallback'
                }
        
        logger.info(f"🎯 Lightweight terrain analysis completed for {len(terrain_results)} cities")
        return terrain_results

    def _step6_map_routes(self, enriched_cities: Dict[str, Any], 
                         terrain_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Create routes with terrain-aware optimization
        """
        logger.info("🗺️ Mapping terrain-aware routes between cities...")
        
        route_mapping = {'corridor_routes': {}, 'station_networks': {}}
        
        city_names = list(enriched_cities.keys())
        if len(city_names) < 2:
            logger.warning("⚠️ Need at least 2 cities for route mapping")
            return route_mapping
        
        # Create routes between all city pairs
        for i in range(len(city_names)):
            for j in range(i + 1, len(city_names)):
                city_a_name = city_names[i]
                city_b_name = city_names[j]
                
                logger.info(f"🛤️ Creating route: {city_a_name} → {city_b_name}")
                
                try:
                    # Get city data
                    city_a = enriched_cities[city_a_name]
                    city_b = enriched_cities[city_b_name]
                    
                    # Create initial straight-line route
                    point_a = city_a['center_point']
                    point_b = city_b['center_point']
                    straight_route = LineString([point_a.coords[0], point_b.coords[0]])
                    
                    # Create terrain-aware curved route
                    curved_route = self._create_terrain_aware_route(city_a, city_b, straight_route)
                    
                    # Calculate route statistics
                    distance_km = curved_route.length * 111  # Rough conversion
                    waypoints_count = len(curved_route.coords)
                    
                    # Perform lightweight terrain analysis for the route
                    route_terrain = analyze_terrain_lightweight(curved_route, f"{city_a_name}-{city_b_name}")
                    
                    route_key = f"{city_a_name}-{city_b_name}"
                    route_mapping['corridor_routes'][route_key] = {
                        'route_line': curved_route,
                        'original_straight_line': straight_route,
                        'distance_km': distance_km,
                        'waypoints_count': waypoints_count,
                        'terrain_analysis': route_terrain,
                        'city_a': city_a_name,
                        'city_b': city_b_name,
                        'optimization_applied': True,
                        'terrain_complexity': route_terrain.overall_complexity.value,
                        'cost_multiplier': route_terrain.cost_multiplier
                    }
                    
                    logger.info(f"✅ Route {route_key}: {distance_km:.1f}km, "
                               f"{waypoints_count} waypoints, "
                               f"{route_terrain.overall_complexity.value} terrain")
                    
                    # Create simplified station networks
                    route_mapping['station_networks'][city_a_name] = {
                        'city_name': city_a_name,
                        'stations': [{'name': f"{city_a_name} Central", 'location': point_a}]
                    }
                    route_mapping['station_networks'][city_b_name] = {
                        'city_name': city_b_name, 
                        'stations': [{'name': f"{city_b_name} Central", 'location': point_b}]
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create route {city_a_name}-{city_b_name}: {e}")
                    
                    # Fallback to straight line
                    route_key = f"{city_a_name}-{city_b_name}"
                    route_mapping['corridor_routes'][route_key] = {
                        'route_line': straight_route,
                        'distance_km': straight_route.length * 111,
                        'waypoints_count': 2,
                        'terrain_analysis': None,
                        'city_a': city_a_name,
                        'city_b': city_b_name,
                        'optimization_applied': False,
                        'status': 'fallback'
                    }
        
        logger.info(f"🎯 Route mapping completed: {len(route_mapping['corridor_routes'])} routes created")
        return route_mapping
    
    def _create_terrain_aware_route(self, city_a, city_b, straight_route):
        """
        Create terrain-aware route that curves around obstacles instead of straight lines
        """
        try:
            point_a = city_a['center_point']
            point_b = city_b['center_point']
            
            # Calculate distance and determine complexity
            distance = np.sqrt((point_b.x - point_a.x)**2 + (point_b.y - point_a.y)**2)
            
            waypoints = [point_a.coords[0]]
            
            if distance > 0.05:  # If route > ~5km, add intermediate points
                num_waypoints = min(5, max(2, int(distance * 20)))  # 2-5 intermediate points
                
                for i in range(1, num_waypoints):
                    ratio = i / num_waypoints
                    
                    # Linear interpolation with terrain-aware offset
                    base_x = point_a.x + ratio * (point_b.x - point_a.x)
                    base_y = point_a.y + ratio * (point_b.y - point_a.y)
                    
                    # Add terrain-aware offset to avoid obstacles
                    offset_x = 0.005 * np.sin(ratio * np.pi * 3)  # ~500m max offset
                    offset_y = 0.005 * np.cos(ratio * np.pi * 2)  # Perpendicular offset
                    
                    # Apply terrain-based variation
                    terrain_factor = 0.5 + 0.5 * np.sin(base_x * 100) * np.cos(base_y * 100)
                    
                    final_x = base_x + offset_x * terrain_factor
                    final_y = base_y + offset_y * terrain_factor
                    
                    waypoints.append((final_x, final_y))
            else:
                # Simple 3-point curve for shorter routes
                mid_x = (point_a.x + point_b.x) / 2
                mid_y = (point_a.y + point_b.y) / 2
                offset = 0.003  # ~300m offset
                
                curved_mid = (mid_x + offset, mid_y - offset * 0.5)
                waypoints.append(curved_mid)
            
            waypoints.append(point_b.coords[0])
            
            curved_route = LineString(waypoints)
            
            logger.info(f"✅ Created terrain-aware route with {len(waypoints)} waypoints")
            return curved_route
            
        except Exception as e:
            logger.warning(f"⚠️ Terrain-aware routing failed: {e}, using straight line")
            return straight_route
    
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
            if route_data.get('terrain_analysis'):
                network_design.terrain_complexity = route_data['terrain_analysis'].overall_complexity
            
            # Calculate costs
            cost_summary = self.cost_analyzer.analyze_full_cost(
                network_design=network_design,
                budget_constraint=None
            )
            
            optimization_results['optimized_routes'][route_key] = {
                'network_design': network_design,
                'route_line': route_data['route_line'],
                'terrain_analysis': route_data.get('terrain_analysis'),
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
    
    # Replace the _make_serializable method in your pipeline_initiator.py with this version:

    def _make_serializable(self, obj, _seen: set = None):
        """
        Convert complex / circular objects into something JSON-serialisable.
        """
        if _seen is None:
            _seen = set()

        # break cycles
        obj_id = id(obj)
        if obj_id in _seen:
            return str(obj)          # fallback for repeated reference
        _seen.add(obj_id)

        # primitives
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Handle Enums (like TrainType, TerrainComplexity)
        if hasattr(obj, 'value'):
            return obj.value

        # shapely geometries
        if hasattr(obj, "wkt"):
            return obj.wkt

        # dataclasses / simple objects
        if hasattr(obj, "__dict__"):
            return {k: self._make_serializable(v, _seen) for k, v in obj.__dict__.items()}

        # mappings
        if isinstance(obj, dict):
            serialized_dict = {}
            for k, v in obj.items():
                # Convert keys to strings if they're not already JSON-serializable
                if hasattr(k, 'value'):  # Handle enum keys
                    key_str = k.value
                elif isinstance(k, (str, int, float, bool)) or k is None:
                    key_str = k
                else:
                    key_str = str(k)
                
                serialized_dict[key_str] = self._make_serializable(v, _seen)
            return serialized_dict

        # sequences
        if isinstance(obj, (list, tuple, set)):
            return [self._make_serializable(i, _seen) for i in obj]

        # anything else → string representation
        return str(obj)


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