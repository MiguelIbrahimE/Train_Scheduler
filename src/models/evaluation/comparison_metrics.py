import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

@dataclass
class RouteComparison:
    """Data class for route comparison results"""
    route_id: str
    optimized_cost: float
    baseline_cost: float
    cost_improvement: float
    cost_improvement_pct: float
    distance_km: float
    time_savings_min: float
    population_served: int
    efficiency_score: float

@dataclass
class NetworkComparison:
    """Data class for network comparison results"""
    country: str
    total_routes: int
    avg_cost_improvement: float
    total_cost_savings: float
    avg_time_savings: float
    total_population_served: int
    network_efficiency: float
    connectivity_improvement: float

class ComparisonMetrics:
    """Comprehensive metrics for comparing route optimization results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Baseline calculation methods
        self.baseline_methods = {
            'shortest_path': self._calculate_shortest_path_baseline,
            'existing_routes': self._calculate_existing_routes_baseline,
            'straight_line': self._calculate_straight_line_baseline,
            'random_routes': self._calculate_random_routes_baseline
        }
        
        # Metric weights for overall scoring
        self.metric_weights = {
            'cost_efficiency': 0.30,
            'time_efficiency': 0.25,
            'distance_efficiency': 0.20,
            'population_coverage': 0.15,
            'connectivity': 0.10
        }
    
    def compare_routes(self, optimized_routes: Dict, baseline_routes: Dict,
                      cities_df: pd.DataFrame, method: str = 'shortest_path') -> List[RouteComparison]:
        """Compare optimized routes against baseline routes"""
        
        comparisons = []
        
        for route_id in optimized_routes.keys():
            if route_id not in baseline_routes:
                self.logger.warning(f"Baseline route not found for {route_id}")
                continue
            
            opt_route = optimized_routes[route_id]
            base_route = baseline_routes[route_id]
            
            # Extract route metrics
            opt_cost = opt_route.get('total_cost', 0)
            base_cost = base_route.get('total_cost', 0)
            
            # Calculate improvements
            cost_improvement = base_cost - opt_cost
            cost_improvement_pct = (cost_improvement / base_cost * 100) if base_cost > 0 else 0
            
            # Calculate distance
            distance_km = sum(seg.get('distance_km', 0) for seg in opt_route.get('segments', []))
            
            # Estimate time savings (simplified)
            time_savings_min = self._estimate_time_savings(opt_route, base_route)
            
            # Calculate population served
            population_served = self._calculate_population_served(opt_route, cities_df)
            
            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency_score(opt_route, base_route)
            
            comparison = RouteComparison(
                route_id=route_id,
                optimized_cost=opt_cost,
                baseline_cost=base_cost,
                cost_improvement=cost_improvement,
                cost_improvement_pct=cost_improvement_pct,
                distance_km=distance_km,
                time_savings_min=time_savings_min,
                population_served=population_served,
                efficiency_score=efficiency_score
            )
            
            comparisons.append(comparison)
        
        return comparisons
    
    def _calculate_shortest_path_baseline(self, route_data: Dict, cities_df: pd.DataFrame) -> Dict:
        """Calculate shortest path baseline"""
        
        if not route_data.get('path'):
            return {'total_cost': float('inf'), 'segments': []}
        
        path = route_data['path']
        if len(path) < 2:
            return {'total_cost': float('inf'), 'segments': []}
        
        # Calculate direct distance cost
        start_city = cities_df[cities_df['city'] == path[0]]
        end_city = cities_df[cities_df['city'] == path[-1]]
        
        if start_city.empty or end_city.empty:
            return {'total_cost': float('inf'), 'segments': []}
        
        # Haversine distance
        distance_km = self._haversine_distance(
            start_city.iloc[0]['lat'], start_city.iloc[0]['lon'],
            end_city.iloc[0]['lat'], end_city.iloc[0]['lon']
        )
        
        # Simple cost model: base cost per km
        base_cost_per_km = 10.0  # millions
        total_cost = distance_km * base_cost_per_km
        
        return {
            'total_cost': total_cost,
            'segments': [{'distance_km': distance_km, 'cost': total_cost}]
        }
    
    def _calculate_existing_routes_baseline(self, route_data: Dict, cities_df: pd.DataFrame) -> Dict:
        """Calculate baseline based on existing routes"""
        
        # This would use actual existing railway data
        # For now, use a penalty factor on shortest path
        shortest_baseline = self._calculate_shortest_path_baseline(route_data, cities_df)
        
        # Existing routes are typically 20-30% more expensive due to terrain, curves, etc.
        penalty_factor = 1.25
        
        return {
            'total_cost': shortest_baseline['total_cost'] * penalty_factor,
            'segments': shortest_baseline['segments']
        }
    
    def _calculate_straight_line_baseline(self, route_data: Dict, cities_df: pd.DataFrame) -> Dict:
        """Calculate straight line baseline (theoretical minimum)"""
        
        return self._calculate_shortest_path_baseline(route_data, cities_df)
    
    def _calculate_random_routes_baseline(self, route_data: Dict, cities_df: pd.DataFrame) -> Dict:
        """Calculate random routes baseline"""
        
        # Generate random cost (for testing purposes)
        straight_line = self._calculate_straight_line_baseline(route_data, cities_df)
        
        # Random factor between 1.5 and 3.0
        random_factor = np.random.uniform(1.5, 3.0)
        
        return {
            'total_cost': straight_line['total_cost'] * random_factor,
            'segments': straight_line['segments']
        }
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance in kilometers"""
        
        R = 6371  # Earth's radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _estimate_time_savings(self, optimized_route: Dict, baseline_route: Dict) -> float:
        """Estimate time savings in minutes"""
        
        # Simplified time calculation
        # Assume average speed of 100 km/h for optimized, 80 km/h for baseline
        
        opt_distance = sum(seg.get('distance_km', 0) for seg in optimized_route.get('segments', []))
        base_distance = sum(seg.get('distance_km', 0) for seg in baseline_route.get('segments', []))
        
        opt_time = opt_distance / 100 * 60  # minutes
        base_time = base_distance / 80 * 60  # minutes
        
        return max(0, base_time - opt_time)
    
    def _calculate_population_served(self, route_data: Dict, cities_df: pd.DataFrame) -> int:
        """Calculate total population served by route"""
        
        if not route_data.get('path'):
            return 0
        
        total_population = 0
        for city_name in route_data['path']:
            city_data = cities_df[cities_df['city'] == city_name]
            if not city_data.empty:
                total_population += city_data.iloc[0].get('population', 0)
        
        return total_population
    
    def _calculate_efficiency_score(self, optimized_route: Dict, baseline_route: Dict) -> float:
        """Calculate overall efficiency score (0-1, higher is better)"""
        
        opt_cost = optimized_route.get('total_cost', float('inf'))
        base_cost = baseline_route.get('total_cost', float('inf'))
        
        if base_cost == 0 or opt_cost == float('inf'):
            return 0.0
        
        # Cost efficiency (0-1)
        cost_efficiency = max(0, 1 - (opt_cost / base_cost))
        
        # Distance efficiency (simplified)
        opt_distance = sum(seg.get('distance_km', 0) for seg in optimized_route.get('segments', []))
        base_distance = sum(seg.get('distance_km', 0) for seg in baseline_route.get('segments', []))
        
        if base_distance > 0:
            distance_efficiency = max(0, 1 - (opt_distance / base_distance))
        else:
            distance_efficiency = 0.5
        
        # Combined efficiency
        efficiency_score = (cost_efficiency * 0.7) + (distance_efficiency * 0.3)
        
        return min(1.0, efficiency_score)
    
    def compare_networks(self, countries_results: Dict[str, Dict], 
                        cities_data: Dict[str, pd.DataFrame],
                        baseline_method: str = 'shortest_path') -> List[NetworkComparison]:
        """Compare network-level results across countries"""
        
        network_comparisons = []
        
        for country, results in countries_results.items():
            if country not in cities_data:
                continue
            
            cities_df = cities_data[country]
            optimized_routes = results
            
            # Generate baseline routes
            baseline_routes = {}
            for route_id, route_data in optimized_routes.items():
                baseline_routes[route_id] = self.baseline_methods[baseline_method](route_data, cities_df)
            
            # Calculate route comparisons
            route_comparisons = self.compare_routes(optimized_routes, baseline_routes, cities_df, baseline_method)
            
            if not route_comparisons:
                continue
            
            # Aggregate network metrics
            total_routes = len(route_comparisons)
            avg_cost_improvement = np.mean([comp.cost_improvement_pct for comp in route_comparisons])
            total_cost_savings = sum([comp.cost_improvement for comp in route_comparisons])
            avg_time_savings = np.mean([comp.time_savings_min for comp in route_comparisons])
            total_population_served = sum([comp.population_served for comp in route_comparisons])
            network_efficiency = np.mean([comp.efficiency_score for comp in route_comparisons])
            
            # Calculate connectivity improvement
            connectivity_improvement = self._calculate_connectivity_improvement(
                optimized_routes, baseline_routes, cities_df
            )
            
            network_comparison = NetworkComparison(
                country=country,
                total_routes=total_routes,
                avg_cost_improvement=avg_cost_improvement,
                total_cost_savings=total_cost_savings,
                avg_time_savings=avg_time_savings,
                total_population_served=total_population_served,
                network_efficiency=network_efficiency,
                connectivity_improvement=connectivity_improvement
            )
            
            network_comparisons.append(network_comparison)
        
        return network_comparisons
    
    def _calculate_connectivity_improvement(self, optimized_routes: Dict, 
                                          baseline_routes: Dict, 
                                          cities_df: pd.DataFrame) -> float:
        """Calculate connectivity improvement metric"""
        
        # Simplified connectivity metric based on route efficiency
        if not optimized_routes or not baseline_routes:
            return 0.0
        
        opt_avg_cost = np.mean([route.get('total_cost', 0) for route in optimized_routes.values()])
        base_avg_cost = np.mean([route.get('total_cost', 0) for route in baseline_routes.values()])
        
        if base_avg_cost == 0:
            return 0.0
        
        # Connectivity improvement as inverse of cost ratio
        connectivity_improvement = max(0, 1 - (opt_avg_cost / base_avg_cost))
        
        return connectivity_improvement
    
    def statistical_significance_test(self, optimized_results: List[float], 
                                    baseline_results: List[float]) -> Dict:
        """Test statistical significance of improvements"""
        
        if len(optimized_results) < 2 or len(baseline_results) < 2:
            return {'significant': False, 'p_value': 1.0, 'test': 'insufficient_data'}
        
        # Paired t-test
        if len(optimized_results) == len(baseline_results):
            statistic, p_value = stats.ttest_rel(baseline_results, optimized_results)
            test_type = 'paired_ttest'
        else:
            # Independent t-test
            statistic, p_value = stats.ttest_ind(baseline_results, optimized_results)
            test_type = 'independent_ttest'
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        if len(optimized_results) == len(baseline_results):
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(baseline_results, optimized_results)
                wilcoxon_result = {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p}
            except:
                wilcoxon_result = None
        else:
            wilcoxon_result = None
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(optimized_results) - 1) * np.var(optimized_results, ddof=1) + 
                             (len(baseline_results) - 1) * np.var(baseline_results, ddof=1)) / 
                            (len(optimized_results) + len(baseline_results) - 2))
        
        if pooled_std > 0:
            cohens_d = (np.mean(baseline_results) - np.mean(optimized_results)) / pooled_std
        else:
            cohens_d = 0
        
        return {
            'significant': p_value < 0.05,
            'p_value': p_value,
            'statistic': statistic,
            'test': test_type,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_effect_size(cohens_d),
            'wilcoxon': wilcoxon_result
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def create_comparison_report(self, route_comparisons: List[RouteComparison],
                               network_comparisons: List[NetworkComparison],
                               output_path: str) -> Dict:
        """Create comprehensive comparison report"""
        
        # Route-level statistics
        route_stats = {
            'total_routes': len(route_comparisons),
            'avg_cost_improvement_pct': np.mean([comp.cost_improvement_pct for comp in route_comparisons]),
            'median_cost_improvement_pct': np.median([comp.cost_improvement_pct for comp in route_comparisons]),
            'std_cost_improvement_pct': np.std([comp.cost_improvement_pct for comp in route_comparisons]),
            'best_improvement_pct': max([comp.cost_improvement_pct for comp in route_comparisons]) if route_comparisons else 0,
            'worst_improvement_pct': min([comp.cost_improvement_pct for comp in route_comparisons]) if route_comparisons else 0,
            'routes_with_improvement': sum([1 for comp in route_comparisons if comp.cost_improvement > 0]),
            'avg_time_savings_min': np.mean([comp.time_savings_min for comp in route_comparisons]),
            'avg_efficiency_score': np.mean([comp.efficiency_score for comp in route_comparisons])
        }
        
        # Network-level statistics
        network_stats = {
            'total_countries': len(network_comparisons),
            'avg_network_efficiency': np.mean([comp.network_efficiency for comp in network_comparisons]),
            'total_population_served': sum([comp.total_population_served for comp in network_comparisons]),
            'total_cost_savings': sum([comp.total_cost_savings for comp in network_comparisons]),
            'avg_connectivity_improvement': np.mean([comp.connectivity_improvement for comp in network_comparisons])
        }
        
        # Statistical significance tests
        if route_comparisons:
            cost_improvements = [comp.cost_improvement_pct for comp in route_comparisons]
            baseline_costs = [comp.baseline_cost for comp in route_comparisons]
            optimized_costs = [comp.optimized_cost for comp in route_comparisons]
            
            significance_test = self.statistical_significance_test(optimized_costs, baseline_costs)
        else:
            significance_test = {'significant': False, 'p_value': 1.0, 'test': 'no_data'}
        
        # Create detailed report
        report = {
            'summary': {
                'report_type': 'Route Optimization Comparison',
                'total_routes_analyzed': len(route_comparisons),
                'total_countries_analyzed': len(network_comparisons),
                'overall_improvement_significant': significance_test['significant'],
                'overall_avg_improvement_pct': route_stats['avg_cost_improvement_pct']
            },
            'route_level_analysis': route_stats,
            'network_level_analysis': network_stats,
            'statistical_analysis': significance_test,
            'detailed_route_results': [
                {
                    'route_id': comp.route_id,
                    'cost_improvement_pct': comp.cost_improvement_pct,
                    'cost_improvement_abs': comp.cost_improvement,
                    'efficiency_score': comp.efficiency_score,
                    'time_savings_min': comp.time_savings_min,
                    'distance_km': comp.distance_km,
                    'population_served': comp.population_served
                }
                for comp in route_comparisons
            ],
            'detailed_network_results': [
                {
                    'country': comp.country,
                    'avg_cost_improvement_pct': comp.avg_cost_improvement,
                    'total_cost_savings': comp.total_cost_savings,
                    'network_efficiency': comp.network_efficiency,
                    'connectivity_improvement': comp.connectivity_improvement,
                    'total_population_served': comp.total_population_served
                }
                for comp in network_comparisons
            ]
        }
        
        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save as CSV for easy analysis
        csv_path = output_file.parent / f"{output_file.stem}.csv"
        
        route_df = pd.DataFrame([
            {
                'route_id': comp.route_id,
                'optimized_cost': comp.optimized_cost,
                'baseline_cost': comp.baseline_cost,
                'cost_improvement': comp.cost_improvement,
                'cost_improvement_pct': comp.cost_improvement_pct,
                'distance_km': comp.distance_km,
                'time_savings_min': comp.time_savings_min,
                'population_served': comp.population_served,
                'efficiency_score': comp.efficiency_score
            }
            for comp in route_comparisons
        ])
        
        route_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Comparison report saved to {output_file}")
        self.logger.info(f"Detailed results saved to {csv_path}")
        
        return report
    
    def create_comparison_visualizations(self, route_comparisons: List[RouteComparison],
                                       network_comparisons: List[NetworkComparison],
                                       output_dir: str):
        """Create visualizations for comparison results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Cost improvement distribution
        if route_comparisons:
            plt.figure(figsize=(12, 8))
            
            improvements = [comp.cost_improvement_pct for comp in route_comparisons]
            
            plt.subplot(2, 2, 1)
            plt.hist(improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(improvements), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(improvements):.1f}%')
            plt.xlabel('Cost Improvement (%)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Cost Improvements')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Efficiency vs Distance scatter
            plt.subplot(2, 2, 2)
            distances = [comp.distance_km for comp in route_comparisons]
            efficiencies = [comp.efficiency_score for comp in route_comparisons]
            
            plt.scatter(distances, efficiencies, alpha=0.6, c=improvements, 
                       cmap='RdYlGn', s=60)
            plt.colorbar(label='Cost Improvement (%)')
            plt.xlabel('Distance (km)')
            plt.ylabel('Efficiency Score')
            plt.title('Efficiency vs Distance')
            plt.grid(True, alpha=0.3)
            
            # 3. Time savings vs Population served
            plt.subplot(2, 2, 3)
            time_savings = [comp.time_savings_min for comp in route_comparisons]
            population_served = [comp.population_served for comp in route_comparisons]
            
            plt.scatter(population_served, time_savings, alpha=0.6, c=improvements, 
                       cmap='RdYlGn', s=60)
            plt.colorbar(label='Cost Improvement (%)')
            plt.xlabel('Population Served')
            plt.ylabel('Time Savings (minutes)')
            plt.title('Time Savings vs Population Served')
            plt.grid(True, alpha=0.3)
            
            # 4. Network comparison by country
            plt.subplot(2, 2, 4)
            if network_comparisons:
                countries = [comp.country for comp in network_comparisons]
                network_efficiencies = [comp.network_efficiency for comp in network_comparisons]
                
                bars = plt.bar(countries, network_efficiencies, alpha=0.7, color='lightcoral')
                plt.xlabel('Country')
                plt.ylabel('Network Efficiency')
                plt.title('Network Efficiency by Country')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, efficiency in zip(bars, network_efficiencies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{efficiency:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'comparison_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Detailed comparison table visualization
        if route_comparisons:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Create comparison table
            comparison_data = []
            for comp in route_comparisons[:10]:  # Show top 10
                comparison_data.append([
                    comp.route_id,
                    f"{comp.cost_improvement_pct:.1f}%",
                    f"{comp.time_savings_min:.0f} min",
                    f"{comp.distance_km:.0f} km",
                    f"{comp.efficiency_score:.2f}",
                    f"{comp.population_served:,}"
                ])
            
            columns = ['Route', 'Cost Improvement', 'Time Savings', 'Distance', 'Efficiency', 'Population']
            
            # Create table
            table = ax.table(cellText=comparison_data, colLabels=columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            
            # Style the table
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code the improvement column
            for i in range(1, len(comparison_data) + 1):
                improvement = route_comparisons[i-1].cost_improvement_pct
                if improvement > 10:
                    color = '#90EE90'  # Light green
                elif improvement > 5:
                    color = '#FFE4B5'  # Light yellow
                else:
                    color = '#FFB6C1'  # Light pink
                table[(i, 1)].set_facecolor(color)
            
            ax.axis('off')
            plt.title('Top 10 Route Optimization Results', fontsize=16, fontweight='bold', pad=20)
            plt.savefig(output_path / 'top_routes_table.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Comparison visualizations saved to {output_path}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create comparison metrics
    comparator = ComparisonMetrics()
    
    # Test with sample data
    sample_optimized = {
        'route_1': {
            'total_cost': 150.0,
            'path': ['Berlin', 'Munich'],
            'segments': [{'distance_km': 350, 'cost': 150.0}]
        },
        'route_2': {
            'total_cost': 200.0,
            'path': ['Hamburg', 'Frankfurt'],
            'segments': [{'distance_km': 400, 'cost': 200.0}]
        }
    }
    
    sample_cities = pd.DataFrame({
        'city': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt'],
        'lat': [52.5200, 48.1351, 53.5511, 50.1109],
        'lon': [13.4050, 11.5820, 9.9937, 8.6821],
        'population': [3669491, 1471508, 1899160, 753056]
    })
    
    # Generate baseline routes
    baseline_routes = {}
    for route_id, route_data in sample_optimized.items():
        baseline_routes[route_id] = comparator._calculate_shortest_path_baseline(route_data, sample_cities)
    
    # Compare routes
    route_comparisons = comparator.compare_routes(sample_optimized, baseline_routes, sample_cities)
    
    # Compare networks
    countries_results = {'germany': sample_optimized}
    cities_data = {'germany': sample_cities}
    network_comparisons = comparator.compare_networks(countries_results, cities_data)
    
    # Create report
    report = comparator.create_comparison_report(
        route_comparisons, network_comparisons, 
        'data/output/reports/comparison_report.json'
    )
    
    # Create visualizations
    comparator.create_comparison_visualizations(
        route_comparisons, network_comparisons,
        'data/output/visualizations/comparisons'
    )
    
    print("=== Route Comparison Results ===")
    for comp in route_comparisons:
        print(f"Route: {comp.route_id}")
        print(f"  Cost Improvement: {comp.cost_improvement_pct:.1f}%")
        print(f"  Efficiency Score: {comp.efficiency_score:.2f}")
        print(f"  Time Savings: {comp.time_savings_min:.0f} minutes")
        print()
    
    print("=== Network Comparison Results ===")
    for comp in network_comparisons:
        print(f"Country: {comp.country}")
        print(f"  Network Efficiency: {comp.network_efficiency:.2f}")
        print(f"  Total Cost Savings: ${comp.total_cost_savings:.1f}M")
        print(f"  Population Served: {comp.total_population_served:,}")
        print()
    
    print(f"Report saved with {len(route_comparisons)} route comparisons and {len(network_comparisons)} network comparisons")