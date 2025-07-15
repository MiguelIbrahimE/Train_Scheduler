import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class RoutePerformanceMetrics:
    """Data class for route performance metrics"""
    route_id: str
    optimization_time_seconds: float
    distance_km: float
    cost_millions: float
    efficiency_score: float
    connectivity_score: float
    terrain_difficulty: float
    population_served: int
    ridership_potential: float
    environmental_impact: float
    construction_feasibility: float
    operational_efficiency: float
    safety_score: float
    accessibility_score: float
    innovation_score: float
    overall_performance: float

@dataclass
class OptimizationPerformanceMetrics:
    """Data class for optimization algorithm performance"""
    algorithm_name: str
    total_routes_processed: int
    avg_optimization_time: float
    success_rate: float
    convergence_rate: float
    solution_quality_score: float
    memory_usage_mb: float
    scalability_score: float
    robustness_score: float
    computational_efficiency: float

@dataclass
class NetworkPerformanceMetrics:
    """Data class for network-level performance metrics"""
    network_id: str
    total_routes: int
    total_length_km: float
    avg_route_performance: float
    network_connectivity: float
    coverage_efficiency: float
    cost_effectiveness: float
    service_quality: float
    sustainability_score: float
    resilience_score: float
    integration_score: float

class PerformanceMetrics:
    """Comprehensive performance evaluation for railway route optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds
        self.performance_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5,
            'poor': 0.3
        }
        
        # Metric weights for overall performance calculation
        self.metric_weights = {
            'cost_efficiency': 0.25,
            'distance_efficiency': 0.20,
            'connectivity': 0.15,
            'feasibility': 0.15,
            'environmental': 0.10,
            'safety': 0.10,
            'accessibility': 0.05
        }
        
        # Benchmark values for normalization
        self.benchmark_values = {
            'cost_per_km': 15.0,  # millions USD
            'optimization_time': 60.0,  # seconds
            'efficiency_score': 0.8,
            'connectivity_score': 0.7,
            'ridership_potential': 1000000,  # annual passengers
            'environmental_impact': 0.3,  # normalized score
            'safety_score': 0.9
        }
    
    def calculate_route_performance(self, route_data: Dict, optimization_time: float = 0,
                                  terrain_data: Dict = None, 
                                  population_data: Dict = None) -> RoutePerformanceMetrics:
        """Calculate comprehensive performance metrics for a route"""
        
        route_id = route_data.get('route_id', 'unknown')
        
        # Basic route metrics
        distance_km = sum(seg.get('distance_km', 0) for seg in route_data.get('segments', []))
        cost_millions = route_data.get('total_cost', 0)
        
        # Calculate individual performance components
        efficiency_score = self._calculate_efficiency_score(route_data)
        connectivity_score = self._calculate_connectivity_score(route_data)
        terrain_difficulty = self._calculate_terrain_difficulty(route_data, terrain_data)
        population_served = self._calculate_population_served(route_data, population_data)
        ridership_potential = self._calculate_ridership_potential(route_data, population_data)
        environmental_impact = self._calculate_environmental_impact(route_data)
        construction_feasibility = self._calculate_construction_feasibility(route_data)
        operational_efficiency = self._calculate_operational_efficiency(route_data)
        safety_score = self._calculate_safety_score(route_data)
        accessibility_score = self._calculate_accessibility_score(route_data)
        innovation_score = self._calculate_innovation_score(route_data)
        
        # Calculate overall performance
        overall_performance = self._calculate_overall_performance({
            'efficiency_score': efficiency_score,
            'connectivity_score': connectivity_score,
            'environmental_impact': environmental_impact,
            'construction_feasibility': construction_feasibility,
            'operational_efficiency': operational_efficiency,
            'safety_score': safety_score,
            'accessibility_score': accessibility_score
        })
        
        return RoutePerformanceMetrics(
            route_id=route_id,
            optimization_time_seconds=optimization_time,
            distance_km=distance_km,
            cost_millions=cost_millions,
            efficiency_score=efficiency_score,
            connectivity_score=connectivity_score,
            terrain_difficulty=terrain_difficulty,
            population_served=population_served,
            ridership_potential=ridership_potential,
            environmental_impact=environmental_impact,
            construction_feasibility=construction_feasibility,
            operational_efficiency=operational_efficiency,
            safety_score=safety_score,
            accessibility_score=accessibility_score,
            innovation_score=innovation_score,
            overall_performance=overall_performance
        )
    
    def _calculate_efficiency_score(self, route_data: Dict) -> float:
        """Calculate route efficiency score (0-1)"""
        
        segments = route_data.get('segments', [])
        if not segments:
            return 0.0
        
        # Distance efficiency (actual vs straight-line distance)
        total_distance = sum(seg.get('distance_km', 0) for seg in segments)
        
        # Get start and end points
        path = route_data.get('path', [])
        if len(path) < 2:
            return 0.5
        
        # Simplified efficiency calculation
        # In practice, this would use actual coordinates
        straight_line_distance = total_distance * 0.85  # Assume 85% efficiency as baseline
        
        distance_efficiency = min(1.0, straight_line_distance / total_distance) if total_distance > 0 else 0
        
        # Cost efficiency
        cost_per_km = route_data.get('total_cost', 0) / total_distance if total_distance > 0 else 0
        benchmark_cost = self.benchmark_values['cost_per_km']
        cost_efficiency = max(0, min(1.0, 1 - ((cost_per_km - benchmark_cost) / benchmark_cost)))
        
        # Combined efficiency
        efficiency_score = (distance_efficiency * 0.6) + (cost_efficiency * 0.4)
        
        return efficiency_score
    
    def _calculate_connectivity_score(self, route_data: Dict) -> float:
        """Calculate connectivity score based on network integration"""
        
        path = route_data.get('path', [])
        if len(path) < 2:
            return 0.0
        
        # Number of cities connected
        cities_connected = len(path)
        connectivity_base = min(1.0, cities_connected / 5)  # Normalize to max 5 cities
        
        # Route type bonus
        route_type = route_data.get('route_type', 'regional')
        type_bonus = {
            'high_speed': 0.3,
            'intercity': 0.2,
            'regional': 0.1,
            'commuter': 0.05
        }.get(route_type, 0.1)
        
        # Integration with existing network
        integration_score = route_data.get('network_integration', 0.5)
        
        connectivity_score = min(1.0, connectivity_base + type_bonus + (integration_score * 0.3))
        
        return connectivity_score
    
    def _calculate_terrain_difficulty(self, route_data: Dict, terrain_data: Dict = None) -> float:
        """Calculate terrain difficulty score (0-1, higher = more difficult)"""
        
        segments = route_data.get('segments', [])
        if not segments:
            return 0.5
        
        difficulty_scores = []
        
        for segment in segments:
            if terrain_data and segment.get('segment_id') in terrain_data:
                terrain_info = terrain_data[segment['segment_id']]
                gradient = terrain_info.get('max_gradient', 0)
                elevation_range = terrain_info.get('elevation_range', 0)
                
                # Calculate difficulty based on gradient and elevation
                gradient_difficulty = min(1.0, gradient / 0.08)  # 8% is very steep
                elevation_difficulty = min(1.0, elevation_range / 1000)  # 1000m is high
                
                segment_difficulty = (gradient_difficulty + elevation_difficulty) / 2
            else:
                # Default difficulty based on segment properties
                if segment.get('requires_tunnel', False):
                    segment_difficulty = 0.8
                elif segment.get('requires_bridge', False):
                    segment_difficulty = 0.6
                else:
                    segment_difficulty = 0.3
            
            difficulty_scores.append(segment_difficulty)
        
        return np.mean(difficulty_scores) if difficulty_scores else 0.5
    
    def _calculate_population_served(self, route_data: Dict, population_data: Dict = None) -> int:
        """Calculate total population served by the route"""
        
        path = route_data.get('path', [])
        if not path:
            return 0
        
        if population_data:
            total_population = 0
            for city in path:
                if city in population_data:
                    total_population += population_data[city].get('population', 0)
            return total_population
        else:
            # Default population estimates
            return len(path) * 200000  # Assume 200k per city
    
    def _calculate_ridership_potential(self, route_data: Dict, population_data: Dict = None) -> float:
        """Calculate ridership potential based on population and route characteristics"""
        
        population_served = self._calculate_population_served(route_data, population_data)
        
        if population_served == 0:
            return 0
        
        # Base ridership rate (trips per person per year)
        base_rate = 10
        
        # Route type multiplier
        route_type = route_data.get('route_type', 'regional')
        type_multiplier = {
            'high_speed': 1.5,
            'intercity': 1.3,
            'regional': 1.0,
            'commuter': 2.0
        }.get(route_type, 1.0)
        
        # Distance factor (optimal distance for ridership)
        distance_km = sum(seg.get('distance_km', 0) for seg in route_data.get('segments', []))
        if distance_km > 0:
            if distance_km < 50:
                distance_factor = 1.2  # Short trips
            elif distance_km < 200:
                distance_factor = 1.5  # Medium trips
            elif distance_km < 500:
                distance_factor = 1.3  # Long trips
            else:
                distance_factor = 1.0  # Very long trips
        else:
            distance_factor = 1.0
        
        ridership_potential = population_served * base_rate * type_multiplier * distance_factor
        
        return ridership_potential
    
    def _calculate_environmental_impact(self, route_data: Dict) -> float:
        """Calculate environmental impact score (0-1, lower = better)"""
        
        segments = route_data.get('segments', [])
        if not segments:
            return 0.5
        
        impact_scores = []
        
        for segment in segments:
            segment_impact = 0.3  # Base impact
            
            # Terrain impact
            if segment.get('requires_tunnel', False):
                segment_impact += 0.1  # Tunnels have lower surface impact
            elif segment.get('requires_bridge', False):
                segment_impact += 0.2  # Bridges have moderate impact
            else:
                segment_impact += 0.3  # Surface routes have higher impact
            
            # Urban vs rural impact
            if segment.get('urban_area', False):
                segment_impact += 0.1  # Urban areas have higher impact
            
            # Protected areas
            if segment.get('protected_area', False):
                segment_impact += 0.2
            
            impact_scores.append(min(1.0, segment_impact))
        
        return np.mean(impact_scores) if impact_scores else 0.5
    
    def _calculate_construction_feasibility(self, route_data: Dict) -> float:
        """Calculate construction feasibility score (0-1, higher = more feasible)"""
        
        segments = route_data.get('segments', [])
        if not segments:
            return 0.5
        
        feasibility_scores = []
        
        for segment in segments:
            feasibility = 0.8  # Base feasibility
            
            # Terrain challenges
            if segment.get('requires_tunnel', False):
                tunnel_length = segment.get('tunnel_length_km', 0)
                if tunnel_length > 10:
                    feasibility -= 0.3
                elif tunnel_length > 5:
                    feasibility -= 0.2
                else:
                    feasibility -= 0.1
            
            if segment.get('requires_bridge', False):
                bridge_length = segment.get('bridge_length_km', 0)
                if bridge_length > 5:
                    feasibility -= 0.2
                elif bridge_length > 2:
                    feasibility -= 0.1
            
            # Land acquisition challenges
            if segment.get('urban_area', False):
                feasibility -= 0.1
            
            if segment.get('protected_area', False):
                feasibility -= 0.2
            
            feasibility_scores.append(max(0, feasibility))
        
        return np.mean(feasibility_scores) if feasibility_scores else 0.5
    
    def _calculate_operational_efficiency(self, route_data: Dict) -> float:
        """Calculate operational efficiency score (0-1)"""
        
        # Route characteristics that affect operations
        distance_km = sum(seg.get('distance_km', 0) for seg in route_data.get('segments', []))
        
        # Optimal distance for operations
        if distance_km > 0:
            if distance_km < 50:
                distance_score = 0.8  # Short routes
            elif distance_km < 200:
                distance_score = 1.0  # Optimal distance
            elif distance_km < 500:
                distance_score = 0.9  # Good distance
            else:
                distance_score = 0.7  # Very long routes
        else:
            distance_score = 0.5
        
        # Electrification bonus
        electrification_bonus = 0.1 if route_data.get('electrified', False) else 0
        
        # Route complexity
        segments = route_data.get('segments', [])
        complexity_penalty = 0
        for segment in segments:
            if segment.get('requires_tunnel', False):
                complexity_penalty += 0.05
            if segment.get('requires_bridge', False):
                complexity_penalty += 0.03
        
        operational_efficiency = min(1.0, distance_score + electrification_bonus - complexity_penalty)
        
        return max(0, operational_efficiency)
    
    def _calculate_safety_score(self, route_data: Dict) -> float:
        """Calculate safety score (0-1, higher = safer)"""
        
        segments = route_data.get('segments', [])
        if not segments:
            return 0.8
        
        safety_scores = []
        
        for segment in segments:
            safety = 0.9  # Base safety
            
            # Grade separations improve safety
            if segment.get('requires_tunnel', False) or segment.get('requires_bridge', False):
                safety += 0.05
            
            # Urban areas have more safety challenges
            if segment.get('urban_area', False):
                safety -= 0.05
            
            # Terrain challenges
            terrain_type = segment.get('terrain_type', 'flat')
            if terrain_type == 'mountainous':
                safety -= 0.1
            elif terrain_type == 'hilly':
                safety -= 0.05
            
            safety_scores.append(min(1.0, max(0, safety)))
        
        return np.mean(safety_scores) if safety_scores else 0.8
    
    def _calculate_accessibility_score(self, route_data: Dict) -> float:
        """Calculate accessibility score (0-1, higher = more accessible)"""
        
        path = route_data.get('path', [])
        if not path:
            return 0.5
        
        # Number of stations affects accessibility
        major_stations = route_data.get('major_stations', 2)
        minor_stations = route_data.get('minor_stations', 0)
        total_stations = major_stations + minor_stations
        
        distance_km = sum(seg.get('distance_km', 0) for seg in route_data.get('segments', []))
        
        if distance_km > 0:
            station_density = total_stations / distance_km * 100  # Stations per 100km
            accessibility_base = min(1.0, station_density / 10)  # 10 stations per 100km is good
        else:
            accessibility_base = 0.5
        
        # Urban connections improve accessibility
        urban_connections = sum(1 for seg in route_data.get('segments', []) if seg.get('urban_area', False))
        urban_bonus = min(0.2, urban_connections * 0.1)
        
        accessibility_score = min(1.0, accessibility_base + urban_bonus)
        
        return accessibility_score
    
    def _calculate_innovation_score(self, route_data: Dict) -> float:
        """Calculate innovation score based on advanced features (0-1)"""
        
        innovation_score = 0.3  # Base score
        
        # Advanced route types
        route_type = route_data.get('route_type', 'regional')
        if route_type == 'high_speed':
            innovation_score += 0.3
        elif route_type == 'intercity':
            innovation_score += 0.2
        
        # Technology features
        if route_data.get('electrified', False):
            innovation_score += 0.1
        
        if route_data.get('automated_systems', False):
            innovation_score += 0.1
        
        if route_data.get('renewable_energy', False):
            innovation_score += 0.1
        
        # Advanced infrastructure
        segments = route_data.get('segments', [])
        for segment in segments:
            if segment.get('smart_infrastructure', False):
                innovation_score += 0.05
            if segment.get('noise_barriers', False):
                innovation_score += 0.02
        
        return min(1.0, innovation_score)
    
    def _calculate_overall_performance(self, metrics: Dict) -> float:
        """Calculate overall performance score"""
        
        weighted_score = 0
        
        # Apply weights to different metrics
        weighted_score += metrics.get('efficiency_score', 0) * self.metric_weights['cost_efficiency']
        weighted_score += metrics.get('connectivity_score', 0) * self.metric_weights['connectivity']
        weighted_score += (1 - metrics.get('environmental_impact', 0.5)) * self.metric_weights['environmental']
        weighted_score += metrics.get('construction_feasibility', 0) * self.metric_weights['feasibility']
        weighted_score += metrics.get('operational_efficiency', 0) * self.metric_weights['distance_efficiency']
        weighted_score += metrics.get('safety_score', 0) * self.metric_weights['safety']
        weighted_score += metrics.get('accessibility_score', 0) * self.metric_weights['accessibility']
        
        return min(1.0, max(0, weighted_score))
    
    def evaluate_optimization_algorithm(self, algorithm_name: str, 
                                      optimization_results: List[Dict],
                                      optimization_times: List[float],
                                      target_solutions: List[Dict] = None) -> OptimizationPerformanceMetrics:
        """Evaluate optimization algorithm performance"""
        
        total_routes = len(optimization_results)
        
        if total_routes == 0:
            return self._create_empty_optimization_metrics(algorithm_name)
        
        # Calculate basic metrics
        avg_optimization_time = np.mean(optimization_times)
        
        # Success rate (routes that found valid solutions)
        successful_routes = sum(1 for result in optimization_results if result.get('path'))
        success_rate = successful_routes / total_routes
        
        # Solution quality score
        quality_scores = []
        for result in optimization_results:
            if result.get('path'):
                # Quality based on cost efficiency and route characteristics
                cost = result.get('total_cost', float('inf'))
                distance = sum(seg.get('distance_km', 0) for seg in result.get('segments', []))
                
                if distance > 0 and cost != float('inf'):
                    cost_per_km = cost / distance
                    quality = max(0, 1 - (cost_per_km / self.benchmark_values['cost_per_km']))
                else:
                    quality = 0
                
                quality_scores.append(quality)
        
        solution_quality_score = np.mean(quality_scores) if quality_scores else 0
        
        # Convergence rate (simplified - based on optimization time vs quality)
        convergence_scores = []
        for i, (result, opt_time) in enumerate(zip(optimization_results, optimization_times)):
            if result.get('path') and opt_time > 0:
                # Better solutions in less time = better convergence
                quality = quality_scores[i] if i < len(quality_scores) else 0
                time_factor = min(1.0, self.benchmark_values['optimization_time'] / opt_time)
                convergence = quality * time_factor
                convergence_scores.append(convergence)
        
        convergence_rate = np.mean(convergence_scores) if convergence_scores else 0
        
        # Memory usage (estimated based on route complexity)
        avg_route_complexity = np.mean([
            len(result.get('segments', [])) for result in optimization_results
        ])
        memory_usage_mb = avg_route_complexity * 10  # Simplified estimation
        
        # Scalability score (performance vs problem size)
        scalability_score = self._calculate_scalability_score(optimization_results, optimization_times)
        
        # Robustness score (consistency across different routes)
        robustness_score = self._calculate_robustness_score(quality_scores, optimization_times)
        
        # Computational efficiency
        computational_efficiency = self._calculate_computational_efficiency(
            optimization_times, quality_scores
        )
        
        return OptimizationPerformanceMetrics(
            algorithm_name=algorithm_name,
            total_routes_processed=total_routes,
            avg_optimization_time=avg_optimization_time,
            success_rate=success_rate,
            convergence_rate=convergence_rate,
            solution_quality_score=solution_quality_score,
            memory_usage_mb=memory_usage_mb,
            scalability_score=scalability_score,
            robustness_score=robustness_score,
            computational_efficiency=computational_efficiency
        )
    
    def _create_empty_optimization_metrics(self, algorithm_name: str) -> OptimizationPerformanceMetrics:
        """Create empty optimization metrics"""
        return OptimizationPerformanceMetrics(
            algorithm_name=algorithm_name,
            total_routes_processed=0,
            avg_optimization_time=0.0,
            success_rate=0.0,
            convergence_rate=0.0,
            solution_quality_score=0.0,
            memory_usage_mb=0.0,
            scalability_score=0.0,
            robustness_score=0.0,
            computational_efficiency=0.0
        )
    
    def _calculate_scalability_score(self, optimization_results: List[Dict], 
                                   optimization_times: List[float]) -> float:
        """Calculate scalability score based on performance vs problem size"""
        
        if not optimization_results:
            return 0.0
        
        # Problem size based on route complexity
        problem_sizes = []
        for result in optimization_results:
            segments = result.get('segments', [])
            cities = result.get('path', [])
            complexity = len(segments) + len(cities)
            problem_sizes.append(complexity)
        
        if len(set(problem_sizes)) < 2:
            return 0.8  # Default score if no size variation
        
        # Calculate correlation between problem size and optimization time
        if len(problem_sizes) > 1 and len(optimization_times) > 1:
            correlation, _ = stats.pearsonr(problem_sizes, optimization_times)
            
            # Good scalability = low correlation between size and time
            scalability_score = max(0, 1 - abs(correlation))
        else:
            scalability_score = 0.5
        
        return scalability_score
    
    def _calculate_robustness_score(self, quality_scores: List[float], 
                                  optimization_times: List[float]) -> float:
        """Calculate robustness score based on consistency"""
        
        if not quality_scores or not optimization_times:
            return 0.0
        
        # Consistency in quality
        quality_std = np.std(quality_scores) if len(quality_scores) > 1 else 0
        quality_consistency = max(0, 1 - (quality_std / np.mean(quality_scores))) if np.mean(quality_scores) > 0 else 0
        
        # Consistency in time
        time_std = np.std(optimization_times) if len(optimization_times) > 1 else 0
        time_consistency = max(0, 1 - (time_std / np.mean(optimization_times))) if np.mean(optimization_times) > 0 else 0
        
        # Combined robustness score
        robustness_score = (quality_consistency * 0.6) + (time_consistency * 0.4)
        
        return robustness_score
    
    def _calculate_computational_efficiency(self, optimization_times: List[float], 
                                         quality_scores: List[float]) -> float:
        """Calculate computational efficiency (quality per unit time)"""
        
        if not optimization_times or not quality_scores:
            return 0.0
        
        # Efficiency = quality / time
        efficiencies = []
        for quality, time in zip(quality_scores, optimization_times):
            if time > 0:
                efficiency = quality / time
                efficiencies.append(efficiency)
        
        if not efficiencies:
            return 0.0
        
        # Normalize against benchmark
        avg_efficiency = np.mean(efficiencies)
        benchmark_efficiency = self.benchmark_values['efficiency_score'] / self.benchmark_values['optimization_time']
        
        computational_efficiency = min(1.0, avg_efficiency / benchmark_efficiency)
        
        return computational_efficiency
    
    def evaluate_network_performance(self, network_routes: List[RoutePerformanceMetrics],
                                   network_id: str = "default") -> NetworkPerformanceMetrics:
        """Evaluate network-level performance metrics"""
        
        if not network_routes:
            return self._create_empty_network_metrics(network_id)
        
        # Basic network statistics
        total_routes = len(network_routes)
        total_length_km = sum(route.distance_km for route in network_routes)
        avg_route_performance = np.mean([route.overall_performance for route in network_routes])
        
        # Network connectivity (based on route interconnections)
        network_connectivity = self._calculate_network_connectivity(network_routes)
        
        # Coverage efficiency (population served vs network length)
        total_population = sum(route.population_served for route in network_routes)
        coverage_efficiency = total_population / (total_length_km + 1) if total_length_km > 0 else 0
        
        # Cost effectiveness (performance vs cost)
        total_cost = sum(route.cost_millions for route in network_routes)
        cost_effectiveness = avg_route_performance / (total_cost / total_routes) if total_cost > 0 else 0
        
        # Service quality (based on accessibility and connectivity)
        service_quality = np.mean([route.accessibility_score for route in network_routes])
        
        # Sustainability score (environmental and operational)
        sustainability_score = np.mean([
            (1 - route.environmental_impact) * 0.6 + route.operational_efficiency * 0.4
            for route in network_routes
        ])
        
        # Resilience score (network robustness)
        resilience_score = self._calculate_network_resilience(network_routes)
        
        # Integration score (how well routes work together)
        integration_score = self._calculate_network_integration(network_routes)
        
        return NetworkPerformanceMetrics(
            network_id=network_id,
            total_routes=total_routes,
            total_length_km=total_length_km,
            avg_route_performance=avg_route_performance,
            network_connectivity=network_connectivity,
            coverage_efficiency=coverage_efficiency,
            cost_effectiveness=cost_effectiveness,
            service_quality=service_quality,
            sustainability_score=sustainability_score,
            resilience_score=resilience_score,
            integration_score=integration_score
        )
    
    def _create_empty_network_metrics(self, network_id: str) -> NetworkPerformanceMetrics:
        """Create empty network metrics"""
        return NetworkPerformanceMetrics(
            network_id=network_id,
            total_routes=0,
            total_length_km=0.0,
            avg_route_performance=0.0,
            network_connectivity=0.0,
            coverage_efficiency=0.0,
            cost_effectiveness=0.0,
            service_quality=0.0,
            sustainability_score=0.0,
            resilience_score=0.0,
            integration_score=0.0
        )
    
    def _calculate_network_connectivity(self, network_routes: List[RoutePerformanceMetrics]) -> float:
        """Calculate network connectivity score"""
        
        # Simplified connectivity based on route interconnections
        connectivity_scores = [route.connectivity_score for route in network_routes]
        
        # Network effect: more routes = better connectivity
        route_count_factor = min(1.0, len(network_routes) / 10)  # Normalize to 10 routes
        
        # Average individual connectivity
        avg_connectivity = np.mean(connectivity_scores) if connectivity_scores else 0
        
        # Combined network connectivity
        network_connectivity = (avg_connectivity * 0.7) + (route_count_factor * 0.3)
        
        return network_connectivity
    
    def _calculate_network_resilience(self, network_routes: List[RoutePerformanceMetrics]) -> float:
        """Calculate network resilience score"""
        
        # Resilience based on redundancy and robustness
        
        # Route diversity (different route types)
        route_types = set()  # Would need route type information
        
        # Performance consistency
        performance_scores = [route.overall_performance for route in network_routes]
        performance_std = np.std(performance_scores) if len(performance_scores) > 1 else 0
        consistency_score = max(0, 1 - performance_std) if performance_scores else 0
        
        # Safety scores
        safety_scores = [route.safety_score for route in network_routes]
        avg_safety = np.mean(safety_scores) if safety_scores else 0
        
        # Construction feasibility
        feasibility_scores = [route.construction_feasibility for route in network_routes]
        avg_feasibility = np.mean(feasibility_scores) if feasibility_scores else 0
        
        # Combined resilience
        resilience_score = (consistency_score * 0.4 + avg_safety * 0.3 + avg_feasibility * 0.3)
        
        return resilience_score
    
    def _calculate_network_integration(self, network_routes: List[RoutePerformanceMetrics]) -> float:
        """Calculate network integration score"""
        
        # Integration based on how well routes complement each other
        
        # Performance balance
        performance_scores = [route.overall_performance for route in network_routes]
        performance_balance = 1 - (np.std(performance_scores) / np.mean(performance_scores)) if performance_scores and np.mean(performance_scores) > 0 else 0
        
        # Cost balance
        costs = [route.cost_millions for route in network_routes if route.cost_millions > 0]
        cost_balance = 1 - (np.std(costs) / np.mean(costs)) if costs and np.mean(costs) > 0 else 0
        
        # Innovation distribution
        innovation_scores = [route.innovation_score for route in network_routes]
        innovation_balance = 1 - (np.std(innovation_scores) / np.mean(innovation_scores)) if innovation_scores and np.mean(innovation_scores) > 0 else 0
        
        # Combined integration
        integration_score = (performance_balance * 0.5 + cost_balance * 0.3 + innovation_balance * 0.2)
        
        return max(0, integration_score)
    
    def create_performance_report(self, route_metrics: List[RoutePerformanceMetrics],
                                algorithm_metrics: List[OptimizationPerformanceMetrics] = None,
                                network_metrics: List[NetworkPerformanceMetrics] = None,
                                output_path: str = None) -> Dict:
        """Create comprehensive performance report"""
        
        report = {
            'report_metadata': {
                'generation_time': datetime.now().isoformat(),
                'total_routes_analyzed': len(route_metrics),
                'report_type': 'performance_analysis'
            }
        }
        
        # Route-level analysis
        if route_metrics:
            performance_scores = [rm.overall_performance for rm in route_metrics]
            efficiency_scores = [rm.efficiency_score for rm in route_metrics]
            
            report['route_analysis'] = {
                'summary': {
                    'total_routes': len(route_metrics),
                    'avg_performance': np.mean(performance_scores),
                    'best_performance': max(performance_scores),
                    'worst_performance': min(performance_scores),
                    'performance_std': np.std(performance_scores)
                },
                'performance_distribution': {
                    'excellent': sum(1 for p in performance_scores if p >= self.performance_thresholds['excellent']),
                    'good': sum(1 for p in performance_scores if self.performance_thresholds['good'] <= p < self.performance_thresholds['excellent']),
                    'acceptable': sum(1 for p in performance_scores if self.performance_thresholds['acceptable'] <= p < self.performance_thresholds['good']),
                    'poor': sum(1 for p in performance_scores if p < self.performance_thresholds['acceptable'])
                },
                'top_performers': [
                    {
                        'route_id': rm.route_id,
                        'overall_performance': rm.overall_performance,
                        'efficiency_score': rm.efficiency_score,
                        'cost_millions': rm.cost_millions
                    }
                    for rm in sorted(route_metrics, key=lambda x: x.overall_performance, reverse=True)[:5]
                ],
                'detailed_metrics': [asdict(rm) for rm in route_metrics]
            }
        
        # Algorithm analysis
        if algorithm_metrics:
            report['algorithm_analysis'] = {
                'algorithms_compared': len(algorithm_metrics),
                'best_algorithm': max(algorithm_metrics, key=lambda x: x.solution_quality_score).algorithm_name,
                'algorithm_performance': [asdict(am) for am in algorithm_metrics]
            }
        
        # Network analysis
        if network_metrics:
            report['network_analysis'] = {
                'networks_analyzed': len(network_metrics),
                'avg_network_performance': np.mean([nm.avg_route_performance for nm in network_metrics]),
                'network_details': [asdict(nm) for nm in network_metrics]
            }
        
        # Save report
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance report saved to {output_file}")
        
        return report
    
    def visualize_performance_metrics(self, route_metrics: List[RoutePerformanceMetrics],
                                    algorithm_metrics: List[OptimizationPerformanceMetrics] = None,
                                    output_dir: str = "data/output/visualizations/performance"):
        """Create visualizations for performance metrics"""
        
        if not route_metrics:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Route performance overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance distribution
        performance_scores = [rm.overall_performance for rm in route_metrics]
        axes[0, 0].hist(performance_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(performance_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(performance_scores):.2f}')
        axes[0, 0].set_xlabel('Overall Performance Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Performance Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Performance vs Cost
        costs = [rm.cost_millions for rm in route_metrics]
        axes[0, 1].scatter(costs, performance_scores, alpha=0.6, s=60)
        axes[0, 1].set_xlabel('Cost (Millions USD)')
        axes[0, 1].set_ylabel('Performance Score')
        axes[0, 1].set_title('Performance vs Cost')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance components radar chart (for average route)
        components = ['efficiency_score', 'connectivity_score', 'safety_score', 
                     'accessibility_score', 'operational_efficiency']
        avg_components = [np.mean([getattr(rm, comp) for rm in route_metrics]) for comp in components]
        
        # Spider plot
        angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False)
        avg_components_plot = avg_components + [avg_components[0]]  # Complete the circle
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        axes[1, 0].plot(angles_plot, avg_components_plot, 'o-', linewidth=2, label='Average Route')
        axes[1, 0].fill(angles_plot, avg_components_plot, alpha=0.25)
        axes[1, 0].set_xticks(angles)
        axes[1, 0].set_xticklabels([comp.replace('_', ' ').title() for comp in components])
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title('Average Performance Components')
        axes[1, 0].grid(True)
        
        # Top performers comparison
        top_routes = sorted(route_metrics, key=lambda x: x.overall_performance, reverse=True)[:5]
        route_names = [rm.route_id for rm in top_routes]
        route_performances = [rm.overall_performance for rm in top_routes]
        
        bars = axes[1, 1].bar(route_names, route_performances, alpha=0.7, color='lightgreen')
        axes[1, 1].set_xlabel('Route ID')
        axes[1, 1].set_ylabel('Performance Score')
        axes[1, 1].set_title('Top 5 Performing Routes')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, perf in zip(bars, route_performances):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{perf:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'route_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Algorithm performance comparison (if available)
        if algorithm_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Algorithm comparison
            alg_names = [am.algorithm_name for am in algorithm_metrics]
            quality_scores = [am.solution_quality_score for am in algorithm_metrics]
            optimization_times = [am.avg_optimization_time for am in algorithm_metrics]
            
            # Quality comparison
            bars1 = axes[0, 0].bar(alg_names, quality_scores, alpha=0.7, color='lightblue')
            axes[0, 0].set_ylabel('Solution Quality Score')
            axes[0, 0].set_title('Algorithm Quality Comparison')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Time comparison
            bars2 = axes[0, 1].bar(alg_names, optimization_times, alpha=0.7, color='orange')
            axes[0, 1].set_ylabel('Average Optimization Time (s)')
            axes[0, 1].set_title('Algorithm Speed Comparison')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Quality vs Time scatter
            axes[1, 0].scatter(optimization_times, quality_scores, s=100, alpha=0.7)
            for i, name in enumerate(alg_names):
                axes[1, 0].annotate(name, (optimization_times[i], quality_scores[i]), 
                                  xytext=(5, 5), textcoords='offset points')
            axes[1, 0].set_xlabel('Optimization Time (s)')
            axes[1, 0].set_ylabel('Solution Quality')
            axes[1, 0].set_title('Quality vs Speed Trade-off')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Success rates
            success_rates = [am.success_rate * 100 for am in algorithm_metrics]
            bars3 = axes[1, 1].bar(alg_names, success_rates, alpha=0.7, color='lightgreen')
            axes[1, 1].set_ylabel('Success Rate (%)')
            axes[1, 1].set_title('Algorithm Success Rates')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(output_path / 'algorithm_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Performance visualizations saved to {output_path}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create performance metrics evaluator
    evaluator = PerformanceMetrics()
    
    # Test with sample data
    sample_route_data = {
        'route_id': 'Berlin_Munich',
        'route_type': 'high_speed',
        'total_cost': 150.0,
        'segments': [
            {
                'distance_km': 350,
                'terrain_type': 'hilly',
                'urban_area': False,
                'requires_tunnel': True,
                'requires_bridge': False
            }
        ],
        'path': ['Berlin', 'Munich'],
        'major_stations': 2,
        'minor_stations': 3,
        'electrified': True
    }
    
    # Calculate route performance
    route_performance = evaluator.calculate_route_performance(
        sample_route_data, optimization_time=45.0
    )
    
    print("=== Route Performance Metrics ===")
    print(f"Route: {route_performance.route_id}")
    print(f"Overall Performance: {route_performance.overall_performance:.2f}")
    print(f"Efficiency Score: {route_performance.efficiency_score:.2f}")
    print(f"Safety Score: {route_performance.safety_score:.2f}")
    print(f"Environmental Impact: {route_performance.environmental_impact:.2f}")
    print(f"Innovation Score: {route_performance.innovation_score:.2f}")
    
    # Test algorithm performance
    sample_optimization_results = [sample_route_data] * 3
    sample_optimization_times = [45.0, 52.3, 38.1]
    
    algorithm_performance = evaluator.evaluate_optimization_algorithm(
        "A* Pathfinder", sample_optimization_results, sample_optimization_times
    )
    
    print(f"\n=== Algorithm Performance Metrics ===")
    print(f"Algorithm: {algorithm_performance.algorithm_name}")
    print(f"Success Rate: {algorithm_performance.success_rate:.1%}")
    print(f"Avg Optimization Time: {algorithm_performance.avg_optimization_time:.1f}s")
    print(f"Solution Quality: {algorithm_performance.solution_quality_score:.2f}")
    print(f"Computational Efficiency: {algorithm_performance.computational_efficiency:.2f}")
    
    # Create performance report
    report = evaluator.create_performance_report(
        [route_performance],
        [algorithm_performance],
        output_path='data/output/reports/performance_report.json'
    )
    
    # Create visualizations
    evaluator.visualize_performance_metrics(
        [route_performance],
        [algorithm_performance]
    )
    
    print(f"\nPerformance analysis completed. Report includes {len(report['route_analysis']['detailed_metrics'])} routes.")