import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class CostBreakdown:
    """Data class for detailed cost breakdown"""
    route_id: str
    total_cost_millions: float
    construction_cost: float
    land_acquisition_cost: float
    infrastructure_cost: float
    rolling_stock_cost: float
    operational_cost_annual: float
    maintenance_cost_annual: float
    environmental_cost: float
    terrain_adjustment: float
    urban_adjustment: float
    financing_cost: float
    contingency_cost: float
    cost_per_km: float
    cost_per_passenger: float
    npv_30_years: float
    roi_percentage: float
    payback_years: float

@dataclass
class CostComparison:
    """Data class for cost comparison between routes or methods"""
    baseline_cost: float
    optimized_cost: float
    absolute_savings: float
    percentage_savings: float
    cost_efficiency_ratio: float
    value_for_money_score: float

class CostAnalyzer:
    """Comprehensive cost analysis for railway route optimization"""
    
    def __init__(self, config_path: str = "config/countries_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        
        # Cost parameters by country (in millions USD per km)
        self.cost_parameters = {
            'germany': {
                'base_construction': 15.0,
                'high_speed_multiplier': 2.5,
                'urban_multiplier': 2.0,
                'terrain_flat': 1.0,
                'terrain_hilly': 1.5,
                'terrain_mountainous': 2.5,
                'tunnel_cost_per_km': 50.0,
                'bridge_cost_per_km': 25.0,
                'station_cost_major': 100.0,
                'station_cost_minor': 20.0,
                'rolling_stock_per_km': 2.0,
                'land_acquisition_per_km': 5.0,
                'environmental_per_km': 3.0
            },
            'switzerland': {
                'base_construction': 25.0,
                'high_speed_multiplier': 2.0,
                'urban_multiplier': 1.8,
                'terrain_flat': 1.0,
                'terrain_hilly': 1.8,
                'terrain_mountainous': 3.0,
                'tunnel_cost_per_km': 80.0,
                'bridge_cost_per_km': 35.0,
                'station_cost_major': 120.0,
                'station_cost_minor': 25.0,
                'rolling_stock_per_km': 2.5,
                'land_acquisition_per_km': 8.0,
                'environmental_per_km': 4.0
            },
            'japan': {
                'base_construction': 20.0,
                'high_speed_multiplier': 3.0,
                'urban_multiplier': 2.5,
                'terrain_flat': 1.0,
                'terrain_hilly': 1.6,
                'terrain_mountainous': 2.8,
                'tunnel_cost_per_km': 60.0,
                'bridge_cost_per_km': 30.0,
                'station_cost_major': 150.0,
                'station_cost_minor': 30.0,
                'rolling_stock_per_km': 2.2,
                'land_acquisition_per_km': 10.0,
                'environmental_per_km': 3.5
            },
            'france': {
                'base_construction': 12.0,
                'high_speed_multiplier': 2.8,
                'urban_multiplier': 1.9,
                'terrain_flat': 1.0,
                'terrain_hilly': 1.4,
                'terrain_mountainous': 2.2,
                'tunnel_cost_per_km': 45.0,
                'bridge_cost_per_km': 22.0,
                'station_cost_major': 80.0,
                'station_cost_minor': 18.0,
                'rolling_stock_per_km': 1.8,
                'land_acquisition_per_km': 4.0,
                'environmental_per_km': 2.5
            },
            'netherlands': {
                'base_construction': 8.0,
                'high_speed_multiplier': 2.2,
                'urban_multiplier': 2.2,
                'terrain_flat': 1.0,
                'terrain_hilly': 1.2,
                'terrain_mountainous': 1.5,
                'tunnel_cost_per_km': 40.0,
                'bridge_cost_per_km': 20.0,
                'station_cost_major': 60.0,
                'station_cost_minor': 15.0,
                'rolling_stock_per_km': 1.5,
                'land_acquisition_per_km': 6.0,
                'environmental_per_km': 2.0
            },
            'austria': {
                'base_construction': 18.0,
                'high_speed_multiplier': 2.4,
                'urban_multiplier': 1.8,
                'terrain_flat': 1.0,
                'terrain_hilly': 1.7,
                'terrain_mountainous': 2.8,
                'tunnel_cost_per_km': 70.0,
                'bridge_cost_per_km': 32.0,
                'station_cost_major': 90.0,
                'station_cost_minor': 22.0,
                'rolling_stock_per_km': 2.1,
                'land_acquisition_per_km': 6.0,
                'environmental_per_km': 3.2
            },
            'sweden': {
                'base_construction': 10.0,
                'high_speed_multiplier': 2.3,
                'urban_multiplier': 1.6,
                'terrain_flat': 1.0,
                'terrain_hilly': 1.3,
                'terrain_mountainous': 2.0,
                'tunnel_cost_per_km': 35.0,
                'bridge_cost_per_km': 18.0,
                'station_cost_major': 50.0,
                'station_cost_minor': 12.0,
                'rolling_stock_per_km': 1.4,
                'land_acquisition_per_km': 3.0,
                'environmental_per_km': 2.8
            },
            'denmark': {
                'base_construction': 9.0,
                'high_speed_multiplier': 2.1,
                'urban_multiplier': 1.7,
                'terrain_flat': 1.0,
                'terrain_hilly': 1.2,
                'terrain_mountainous': 1.4,
                'tunnel_cost_per_km': 38.0,
                'bridge_cost_per_km': 19.0,
                'station_cost_major': 45.0,
                'station_cost_minor': 10.0,
                'rolling_stock_per_km': 1.3,
                'land_acquisition_per_km': 4.0,
                'environmental_per_km': 2.2
            }
        }
        
        # Economic parameters
        self.economic_params = {
            'discount_rate': 0.04,  # 4% annual discount rate
            'inflation_rate': 0.02,  # 2% annual inflation
            'project_lifetime': 30,  # 30 years
            'maintenance_rate': 0.03,  # 3% of construction cost annually
            'operational_cost_factor': 0.02,  # 2% of construction cost annually
            'ridership_growth_rate': 0.015,  # 1.5% annual growth
            'fare_per_km': 0.10  # $0.10 per passenger-km
        }
    
    def calculate_detailed_cost_breakdown(self, route_data: Dict, country: str, 
                                        terrain_data: Dict = None,
                                        ridership_data: Dict = None) -> CostBreakdown:
        """Calculate detailed cost breakdown for a route"""
        
        if country not in self.cost_parameters:
            self.logger.warning(f"Cost parameters not found for {country}, using default")
            country = 'germany'  # Default fallback
        
        params = self.cost_parameters[country]
        
        # Extract route information
        route_id = route_data.get('route_id', 'unknown')
        segments = route_data.get('segments', [])
        
        if not segments:
            self.logger.warning(f"No segments found for route {route_id}")
            return self._create_empty_cost_breakdown(route_id)
        
        # Calculate total distance
        total_distance_km = sum(seg.get('distance_km', 0) for seg in segments)
        
        if total_distance_km == 0:
            return self._create_empty_cost_breakdown(route_id)
        
        # Base construction cost
        base_cost = params['base_construction'] * total_distance_km
        
        # Route type adjustments
        route_type = route_data.get('route_type', 'regional')
        if route_type in ['high_speed', 'intercity']:
            base_cost *= params['high_speed_multiplier']
        
        # Terrain adjustments
        terrain_cost = self._calculate_terrain_cost(segments, params, terrain_data)
        
        # Urban adjustments
        urban_cost = self._calculate_urban_cost(segments, params, route_data)
        
        # Infrastructure costs
        infrastructure_cost = self._calculate_infrastructure_cost(route_data, params, total_distance_km)
        
        # Rolling stock cost
        rolling_stock_cost = params['rolling_stock_per_km'] * total_distance_km
        
        # Land acquisition cost
        land_acquisition_cost = params['land_acquisition_per_km'] * total_distance_km
        
        # Environmental mitigation cost
        environmental_cost = params['environmental_per_km'] * total_distance_km
        
        # Construction cost (sum of major components)
        construction_cost = base_cost + terrain_cost + urban_cost + infrastructure_cost
        
        # Financing cost (interest during construction, typically 2-3 years)
        financing_cost = construction_cost * 0.08  # 8% financing cost
        
        # Contingency (typically 15-20% of construction cost)
        contingency_cost = construction_cost * 0.18
        
        # Total project cost
        total_cost = (construction_cost + rolling_stock_cost + land_acquisition_cost + 
                     environmental_cost + financing_cost + contingency_cost)
        
        # Operational costs (annual)
        operational_cost_annual = total_cost * self.economic_params['operational_cost_factor']
        maintenance_cost_annual = construction_cost * self.economic_params['maintenance_rate']
        
        # Economic analysis
        annual_ridership = ridership_data.get('annual_ridership', 1000000) if ridership_data else 1000000
        cost_per_passenger = total_cost / annual_ridership if annual_ridership > 0 else 0
        
        # NPV calculation
        npv_30_years = self._calculate_npv(total_cost, operational_cost_annual + maintenance_cost_annual, 
                                         annual_ridership, total_distance_km)
        
        # ROI calculation
        annual_revenue = annual_ridership * self.economic_params['fare_per_km'] * (total_distance_km / 2)  # Average trip distance
        roi_percentage = ((annual_revenue - operational_cost_annual - maintenance_cost_annual) / total_cost) * 100
        
        # Payback period
        annual_net_cash_flow = annual_revenue - operational_cost_annual - maintenance_cost_annual
        payback_years = total_cost / annual_net_cash_flow if annual_net_cash_flow > 0 else float('inf')
        
        return CostBreakdown(
            route_id=route_id,
            total_cost_millions=total_cost,
            construction_cost=construction_cost,
            land_acquisition_cost=land_acquisition_cost,
            infrastructure_cost=infrastructure_cost,
            rolling_stock_cost=rolling_stock_cost,
            operational_cost_annual=operational_cost_annual,
            maintenance_cost_annual=maintenance_cost_annual,
            environmental_cost=environmental_cost,
            terrain_adjustment=terrain_cost - base_cost,
            urban_adjustment=urban_cost,
            financing_cost=financing_cost,
            contingency_cost=contingency_cost,
            cost_per_km=total_cost / total_distance_km,
            cost_per_passenger=cost_per_passenger,
            npv_30_years=npv_30_years,
            roi_percentage=roi_percentage,
            payback_years=payback_years
        )
    
    def _create_empty_cost_breakdown(self, route_id: str) -> CostBreakdown:
        """Create empty cost breakdown for invalid routes"""
        return CostBreakdown(
            route_id=route_id,
            total_cost_millions=0.0,
            construction_cost=0.0,
            land_acquisition_cost=0.0,
            infrastructure_cost=0.0,
            rolling_stock_cost=0.0,
            operational_cost_annual=0.0,
            maintenance_cost_annual=0.0,
            environmental_cost=0.0,
            terrain_adjustment=0.0,
            urban_adjustment=0.0,
            financing_cost=0.0,
            contingency_cost=0.0,
            cost_per_km=0.0,
            cost_per_passenger=0.0,
            npv_30_years=0.0,
            roi_percentage=0.0,
            payback_years=float('inf')
        )
    
    def _calculate_terrain_cost(self, segments: List[Dict], params: Dict, 
                              terrain_data: Dict = None) -> float:
        """Calculate terrain-related costs"""
        
        terrain_cost = 0.0
        
        for segment in segments:
            distance_km = segment.get('distance_km', 0)
            
            # Get terrain type from terrain data or segment
            if terrain_data and segment.get('segment_id') in terrain_data:
                terrain_type = terrain_data[segment['segment_id']].get('terrain_type', 'flat')
            else:
                terrain_type = segment.get('terrain_type', 'flat')
            
            # Apply terrain multiplier
            terrain_multiplier = params.get(f'terrain_{terrain_type}', 1.0)
            segment_cost = params['base_construction'] * distance_km * terrain_multiplier
            
            # Add special infrastructure costs
            if segment.get('requires_tunnel', False):
                tunnel_length = segment.get('tunnel_length_km', distance_km * 0.1)
                segment_cost += tunnel_length * params['tunnel_cost_per_km']
            
            if segment.get('requires_bridge', False):
                bridge_length = segment.get('bridge_length_km', distance_km * 0.05)
                segment_cost += bridge_length * params['bridge_cost_per_km']
            
            terrain_cost += segment_cost
        
        return terrain_cost
    
    def _calculate_urban_cost(self, segments: List[Dict], params: Dict, 
                            route_data: Dict) -> float:
        """Calculate urban area cost adjustments"""
        
        urban_cost = 0.0
        
        for segment in segments:
            distance_km = segment.get('distance_km', 0)
            
            # Check if segment passes through urban areas
            if segment.get('urban_area', False) or segment.get('population_density', 0) > 1000:
                urban_adjustment = params['base_construction'] * distance_km * (params['urban_multiplier'] - 1)
                urban_cost += urban_adjustment
        
        return urban_cost
    
    def _calculate_infrastructure_cost(self, route_data: Dict, params: Dict, 
                                     total_distance_km: float) -> float:
        """Calculate infrastructure costs (stations, signals, etc.)"""
        
        infrastructure_cost = 0.0
        
        # Station costs
        major_stations = route_data.get('major_stations', 2)  # Start and end
        minor_stations = route_data.get('minor_stations', max(1, int(total_distance_km / 50)))  # One per 50km
        
        infrastructure_cost += major_stations * params['station_cost_major']
        infrastructure_cost += minor_stations * params['station_cost_minor']
        
        # Signaling and control systems (estimated as % of base construction)
        signaling_cost = params['base_construction'] * total_distance_km * 0.15
        infrastructure_cost += signaling_cost
        
        # Power supply systems (for electrified routes)
        if route_data.get('electrified', True):
            power_supply_cost = params['base_construction'] * total_distance_km * 0.12
            infrastructure_cost += power_supply_cost
        
        return infrastructure_cost
    
    def _calculate_npv(self, initial_cost: float, annual_operating_cost: float, 
                      annual_ridership: int, route_distance_km: float) -> float:
        """Calculate Net Present Value over project lifetime"""
        
        discount_rate = self.economic_params['discount_rate']
        project_lifetime = self.economic_params['project_lifetime']
        ridership_growth = self.economic_params['ridership_growth_rate']
        fare_per_km = self.economic_params['fare_per_km']
        
        npv = -initial_cost  # Initial investment
        
        for year in range(1, project_lifetime + 1):
            # Annual ridership with growth
            ridership_year = annual_ridership * (1 + ridership_growth) ** year
            
            # Average trip distance (assumed to be half the route distance)
            avg_trip_distance = route_distance_km / 2
            
            # Annual revenue
            annual_revenue = ridership_year * fare_per_km * avg_trip_distance
            
            # Annual net cash flow
            annual_net_cash_flow = annual_revenue - annual_operating_cost
            
            # Discount to present value
            present_value = annual_net_cash_flow / (1 + discount_rate) ** year
            npv += present_value
        
        return npv
    
    def compare_route_costs(self, route1_cost: CostBreakdown, 
                           route2_cost: CostBreakdown) -> CostComparison:
        """Compare costs between two routes"""
        
        baseline_cost = route1_cost.total_cost_millions
        optimized_cost = route2_cost.total_cost_millions
        
        if baseline_cost == 0:
            return CostComparison(
                baseline_cost=baseline_cost,
                optimized_cost=optimized_cost,
                absolute_savings=0,
                percentage_savings=0,
                cost_efficiency_ratio=0,
                value_for_money_score=0
            )
        
        absolute_savings = baseline_cost - optimized_cost
        percentage_savings = (absolute_savings / baseline_cost) * 100
        
        # Cost efficiency ratio (lower is better)
        cost_efficiency_ratio = optimized_cost / baseline_cost
        
        # Value for money score (considers cost, ROI, and payback)
        value_score = self._calculate_value_for_money_score(route2_cost)
        
        return CostComparison(
            baseline_cost=baseline_cost,
            optimized_cost=optimized_cost,
            absolute_savings=absolute_savings,
            percentage_savings=percentage_savings,
            cost_efficiency_ratio=cost_efficiency_ratio,
            value_for_money_score=value_score
        )
    
    def _calculate_value_for_money_score(self, cost_breakdown: CostBreakdown) -> float:
        """Calculate value for money score (0-100, higher is better)"""
        
        # Normalize components to 0-1 scale
        
        # Cost per km (lower is better, normalize against typical range 5-50M per km)
        cost_per_km_score = max(0, 1 - (cost_breakdown.cost_per_km / 50))
        
        # ROI (higher is better, normalize against typical range 0-15%)
        roi_score = min(1, max(0, cost_breakdown.roi_percentage / 15))
        
        # Payback period (shorter is better, normalize against typical range 10-30 years)
        if cost_breakdown.payback_years == float('inf'):
            payback_score = 0
        else:
            payback_score = max(0, 1 - (cost_breakdown.payback_years / 30))
        
        # NPV (higher is better, but normalize relative to total cost)
        npv_ratio = cost_breakdown.npv_30_years / cost_breakdown.total_cost_millions
        npv_score = min(1, max(0, (npv_ratio + 0.5) / 1.5))  # Normalize around break-even
        
        # Weighted combination
        value_score = (cost_per_km_score * 0.3 + 
                      roi_score * 0.3 + 
                      payback_score * 0.2 + 
                      npv_score * 0.2) * 100
        
        return min(100, max(0, value_score))
    
    def analyze_cost_sensitivity(self, route_data: Dict, country: str,
                               sensitivity_params: Dict = None) -> Dict:
        """Perform cost sensitivity analysis"""
        
        if sensitivity_params is None:
            sensitivity_params = {
                'construction_cost': [-20, -10, 0, 10, 20, 30],  # % changes
                'ridership': [-30, -15, 0, 15, 30, 50],
                'discount_rate': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                'operational_cost': [-10, -5, 0, 5, 10, 20]
            }
        
        # Base case
        base_cost = self.calculate_detailed_cost_breakdown(route_data, country)
        
        sensitivity_results = {
            'base_case': {
                'total_cost': base_cost.total_cost_millions,
                'npv': base_cost.npv_30_years,
                'roi': base_cost.roi_percentage,
                'payback': base_cost.payback_years
            },
            'sensitivity_analysis': {}
        }
        
        for param, values in sensitivity_params.items():
            param_results = []
            
            for value in values:
                # Modify parameters
                modified_route_data = route_data.copy()
                modified_params = self.economic_params.copy()
                
                if param == 'construction_cost':
                    # Modify construction cost multiplier
                    modified_params['construction_cost_multiplier'] = 1 + (value / 100)
                elif param == 'ridership':
                    # Modify ridership
                    ridership_data = {'annual_ridership': 1000000 * (1 + value / 100)}
                    modified_cost = self.calculate_detailed_cost_breakdown(
                        modified_route_data, country, ridership_data=ridership_data
                    )
                elif param == 'discount_rate':
                    # Modify discount rate
                    original_rate = self.economic_params['discount_rate']
                    self.economic_params['discount_rate'] = value
                    modified_cost = self.calculate_detailed_cost_breakdown(modified_route_data, country)
                    self.economic_params['discount_rate'] = original_rate
                elif param == 'operational_cost':
                    # Modify operational cost factor
                    original_factor = self.economic_params['operational_cost_factor']
                    self.economic_params['operational_cost_factor'] = original_factor * (1 + value / 100)
                    modified_cost = self.calculate_detailed_cost_breakdown(modified_route_data, country)
                    self.economic_params['operational_cost_factor'] = original_factor
                
                if param != 'discount_rate':
                    modified_cost = self.calculate_detailed_cost_breakdown(modified_route_data, country)
                
                param_results.append({
                    'parameter_value': value,
                    'total_cost': modified_cost.total_cost_millions,
                    'npv': modified_cost.npv_30_years,
                    'roi': modified_cost.roi_percentage,
                    'payback': modified_cost.payback_years
                })
            
            sensitivity_results['sensitivity_analysis'][param] = param_results
        
        return sensitivity_results
    
    def create_cost_report(self, cost_breakdowns: List[CostBreakdown], 
                          cost_comparisons: List[CostComparison] = None,
                          output_path: str = None) -> Dict:
        """Create comprehensive cost analysis report"""
        
        if not cost_breakdowns:
            return {}
        
        # Aggregate statistics
        total_costs = [cb.total_cost_millions for cb in cost_breakdowns]
        construction_costs = [cb.construction_cost for cb in cost_breakdowns]
        cost_per_km = [cb.cost_per_km for cb in cost_breakdowns]
        roi_values = [cb.roi_percentage for cb in cost_breakdowns if cb.roi_percentage != 0]
        payback_periods = [cb.payback_years for cb in cost_breakdowns if cb.payback_years != float('inf')]
        
        report = {
            'summary': {
                'total_routes_analyzed': len(cost_breakdowns),
                'total_investment_required': sum(total_costs),
                'average_cost_per_route': np.mean(total_costs),
                'average_cost_per_km': np.mean(cost_per_km),
                'cost_range': {
                    'min': min(total_costs),
                    'max': max(total_costs),
                    'std': np.std(total_costs)
                }
            },
            'economic_analysis': {
                'average_roi': np.mean(roi_values) if roi_values else 0,
                'average_payback_years': np.mean(payback_periods) if payback_periods else 0,
                'viable_projects': len([cb for cb in cost_breakdowns if cb.roi_percentage > 5 and cb.payback_years < 25]),
                'total_annual_operational_cost': sum([cb.operational_cost_annual for cb in cost_breakdowns])
            },
            'cost_breakdown_analysis': {
                'avg_construction_percentage': np.mean([cb.construction_cost / cb.total_cost_millions * 100 for cb in cost_breakdowns]),
                'avg_rolling_stock_percentage': np.mean([cb.rolling_stock_cost / cb.total_cost_millions * 100 for cb in cost_breakdowns]),
                'avg_land_acquisition_percentage': np.mean([cb.land_acquisition_cost / cb.total_cost_millions * 100 for cb in cost_breakdowns]),
                'avg_contingency_percentage': np.mean([cb.contingency_cost / cb.total_cost_millions * 100 for cb in cost_breakdowns])
            },
            'detailed_routes': [asdict(cb) for cb in cost_breakdowns]
        }
        
        # Add comparison analysis if available
        if cost_comparisons:
            savings = [cc.absolute_savings for cc in cost_comparisons]
            savings_pct = [cc.percentage_savings for cc in cost_comparisons]
            
            report['comparison_analysis'] = {
                'total_savings': sum(savings),
                'average_savings_percentage': np.mean(savings_pct),
                'routes_with_savings': len([cc for cc in cost_comparisons if cc.absolute_savings > 0]),
                'best_savings_percentage': max(savings_pct) if savings_pct else 0,
                'average_value_for_money_score': np.mean([cc.value_for_money_score for cc in cost_comparisons])
            }
        
        # Save report if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Cost analysis report saved to {output_file}")
        
        return report
    
    def visualize_cost_analysis(self, cost_breakdowns: List[CostBreakdown], 
                              output_dir: str):
        """Create visualizations for cost analysis"""
        
        if not cost_breakdowns:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Cost breakdown by component
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cost components
        components = ['construction_cost', 'rolling_stock_cost', 'land_acquisition_cost', 
                     'environmental_cost', 'financing_cost', 'contingency_cost']
        
        avg_costs = {comp: np.mean([getattr(cb, comp) for cb in cost_breakdowns]) for comp in components}
        
        axes[0, 0].pie(avg_costs.values(), labels=[comp.replace('_', ' ').title() for comp in components], 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Average Cost Breakdown by Component')
        
        # Total cost distribution
        total_costs = [cb.total_cost_millions for cb in cost_breakdowns]
        axes[0, 1].hist(total_costs, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Total Cost (Millions USD)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Total Costs')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ROI vs Cost scatter
        roi_values = [cb.roi_percentage for cb in cost_breakdowns]
        axes[1, 0].scatter(total_costs, roi_values, alpha=0.6, s=60)
        axes[1, 0].set_xlabel('Total Cost (Millions USD)')
        axes[1, 0].set_ylabel('ROI (%)')
        axes[1, 0].set_title('ROI vs Total Cost')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add horizontal line at break-even ROI
        axes[1, 0].axhline(y=4, color='red', linestyle='--', alpha=0.7, label='Break-even (4%)')
        axes[1, 0].legend()
        
        # Cost per km comparison
        cost_per_km = [cb.cost_per_km for cb in cost_breakdowns]
        route_ids = [cb.route_id for cb in cost_breakdowns]
        
        # Show top 10 routes by cost per km
        sorted_indices = np.argsort(cost_per_km)[:10]
        top_routes = [route_ids[i] for i in sorted_indices]
        top_costs = [cost_per_km[i] for i in sorted_indices]
        
        bars = axes[1, 1].bar(range(len(top_routes)), top_costs, alpha=0.7, color='lightcoral')
        axes[1, 1].set_xlabel('Routes')
        axes[1, 1].set_ylabel('Cost per km (Millions USD)')
        axes[1, 1].set_title('Top 10 Most Cost-Efficient Routes')
        axes[1, 1].set_xticks(range(len(top_routes)))
        axes[1, 1].set_xticklabels(top_routes, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, cost in zip(bars, top_costs):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{cost:.1f}M', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'cost_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Economic analysis visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # NPV distribution
        npv_values = [cb.npv_30_years for cb in cost_breakdowns]
        axes[0, 0].hist(npv_values, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        axes[0, 0].set_xlabel('NPV (Millions USD)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Net Present Value Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Payback period distribution
        payback_periods = [cb.payback_years for cb in cost_breakdowns if cb.payback_years != float('inf')]
        if payback_periods:
            axes[0, 1].hist(payback_periods, bins=15, alpha=0.7, color='orange', edgecolor='black')
            axes[0, 1].axvline(x=20, color='red', linestyle='--', alpha=0.7, label='20-year threshold')
            axes[0, 1].set_xlabel('Payback Period (Years)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Payback Period Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # ROI vs NPV scatter
        axes[1, 0].scatter(roi_values, npv_values, alpha=0.6, s=60, c=total_costs, cmap='viridis')
        axes[1, 0].set_xlabel('ROI (%)')
        axes[1, 0].set_ylabel('NPV (Millions USD)')
        axes[1, 0].set_title('ROI vs NPV (Color = Total Cost)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add quadrant lines
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].axvline(x=4, color='black', linestyle='-', alpha=0.3)
        
        # Cost per passenger analysis
        cost_per_passenger = [cb.cost_per_passenger for cb in cost_breakdowns if cb.cost_per_passenger > 0]
        if cost_per_passenger:
            axes[1, 1].boxplot(cost_per_passenger)
            axes[1, 1].set_ylabel('Cost per Passenger (USD)')
            axes[1, 1].set_title('Cost per Passenger Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'economic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Detailed cost breakdown table
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Select top 10 routes by total cost
        sorted_by_cost = sorted(cost_breakdowns, key=lambda x: x.total_cost_millions, reverse=True)[:10]
        
        table_data = []
        for cb in sorted_by_cost:
            table_data.append([
                cb.route_id,
                f"${cb.total_cost_millions:.1f}M",
                f"${cb.construction_cost:.1f}M",
                f"${cb.cost_per_km:.1f}M",
                f"{cb.roi_percentage:.1f}%",
                f"{cb.payback_years:.1f}y" if cb.payback_years != float('inf') else "∞",
                f"${cb.npv_30_years:.1f}M"
            ])
        
        columns = ['Route ID', 'Total Cost', 'Construction', 'Cost/km', 'ROI', 'Payback', 'NPV']
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code the ROI column
        for i in range(1, len(table_data) + 1):
            roi_value = sorted_by_cost[i-1].roi_percentage
            if roi_value > 8:
                color = '#90EE90'  # Light green
            elif roi_value > 4:
                color = '#FFE4B5'  # Light yellow
            else:
                color = '#FFB6C1'  # Light pink
            table[(i, 4)].set_facecolor(color)
        
        ax.axis('off')
        plt.title('Top 10 Routes by Total Cost - Detailed Analysis', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(output_path / 'detailed_cost_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Cost analysis visualizations saved to {output_path}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create cost analyzer
    analyzer = CostAnalyzer()
    
    # Test with sample data
    sample_route_data = {
        'route_id': 'Berlin_Munich',
        'route_type': 'high_speed',
        'segments': [
            {
                'distance_km': 350,
                'terrain_type': 'hilly',
                'urban_area': False,
                'requires_tunnel': True,
                'tunnel_length_km': 20,
                'requires_bridge': True,
                'bridge_length_km': 10
            }
        ],
        'major_stations': 2,
        'minor_stations': 4,
        'electrified': True
    }
    
    # Calculate cost breakdown
    cost_breakdown = analyzer.calculate_detailed_cost_breakdown(sample_route_data, 'germany')
    
    print("=== Detailed Cost Breakdown ===")
    print(f"Route: {cost_breakdown.route_id}")
    print(f"Total Cost: ${cost_breakdown.total_cost_millions:.1f}M")
    print(f"Construction Cost: ${cost_breakdown.construction_cost:.1f}M")
    print(f"Cost per km: ${cost_breakdown.cost_per_km:.1f}M")
    print(f"ROI: {cost_breakdown.roi_percentage:.1f}%")
    print(f"Payback Period: {cost_breakdown.payback_years:.1f} years")
    print(f"NPV (30 years): ${cost_breakdown.npv_30_years:.1f}M")
    
    # Create sensitivity analysis
    sensitivity_results = analyzer.analyze_cost_sensitivity(sample_route_data, 'germany')
    print(f"\n=== Sensitivity Analysis ===")
    print(f"Base NPV: ${sensitivity_results['base_case']['npv']:.1f}M")
    print(f"Base ROI: {sensitivity_results['base_case']['roi']:.1f}%")
    
    # Create report
    report = analyzer.create_cost_report(
        [cost_breakdown], 
        output_path='data/output/reports/cost_analysis_report.json'
    )
    
    # Create visualizations
    analyzer.visualize_cost_analysis(
        [cost_breakdown], 
        'data/output/visualizations/cost_analysis'
    )
    
    print(f"\nCost analysis completed. Report saved with {report['summary']['total_routes_analyzed']} routes analyzed.")
    print(f"Total investment required: ${report['summary']['total_investment_required']:.1f}M")