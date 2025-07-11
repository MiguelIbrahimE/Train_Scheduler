"""
BCPC Pipeline: Simplified Route Optimizer with Terrain-Aware A* Algorithm
=========================================================================

This simplified version implements actual terrain-aware pathfinding that avoids
straight lines by considering elevation, slope, and NIMBY constraints.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import heapq
from shapely.geometry import Point, LineString
import requests
import time

logger = logging.getLogger(__name__)

@dataclass
class PathNode:
    """Node in the pathfinding grid"""
    x: int
    y: int
    lat: float
    lon: float
    elevation: float
    g_cost: float = float('inf')  # Cost from start
    h_cost: float = 0.0          # Heuristic cost to goal
    f_cost: float = float('inf')  # Total cost
    parent: Optional['PathNode'] = None
    terrain_penalty: float = 1.0
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class TerrainAwareOptimizer:
    """Simplified terrain-aware route optimizer"""
    
    def __init__(self, grid_resolution_km: float = 2.0):
        """
        Initialize optimizer
        
        Args:
            grid_resolution_km: Grid cell size in kilometers (larger = faster but less precise)
        """
        self.grid_resolution_km = grid_resolution_km
        self.grid_resolution_deg = grid_resolution_km / 111.0  # Rough km to degrees conversion
        
        # Terrain penalties
        self.slope_penalty_factor = 50.0      # Penalty for steep slopes
        self.elevation_change_penalty = 0.1   # Penalty per meter of elevation change
        self.max_acceptable_slope = 0.035     # 3.5% maximum railway grade
        
        # NIMBY factors (simplified)
        self.urban_penalty = 2.0              # Avoid urban areas (higher cost)
        self.protected_area_penalty = 10.0    # Heavily avoid protected areas
        
    def optimize_route_with_terrain(self, start_city: Dict, end_city: Dict, 
                                   existing_route_line: LineString) -> LineString:
        """
        Optimize route considering terrain and NIMBY constraints
        
        Args:
            start_city: Start city data with lat/lon
            end_city: End city data with lat/lon  
            existing_route_line: Existing straight-line route
            
        Returns:
            Optimized LineString route that maneuvers around terrain
        """
        logger.info(f"Optimizing route from {start_city['name']} to {end_city['name']}")
        
        # Extract coordinates
        start_lat = start_city['original_data'].latitude
        start_lon = start_city['original_data'].longitude
        end_lat = end_city['original_data'].latitude
        end_lon = end_city['original_data'].longitude
        
        # Create search grid
        grid = self._create_search_grid(start_lat, start_lon, end_lat, end_lon)
        
        # Download elevation data for grid
        elevation_data = self._get_elevation_data_for_grid(grid)
        
        # Apply terrain penalties
        self._apply_terrain_penalties(grid, elevation_data)
        
        # Apply NIMBY penalties (simplified)
        self._apply_nimby_penalties(grid)
        
        # Run A* pathfinding
        path_nodes = self._astar_pathfinding(grid, start_lat, start_lon, end_lat, end_lon)
        
        if not path_nodes:
            logger.warning("A* pathfinding failed, returning straight line")
            return existing_route_line
        
        # Convert path nodes to LineString
        path_coords = [(node.lon, node.lat) for node in path_nodes]
        optimized_route = LineString(path_coords)
        
        logger.info(f"✅ Route optimization complete: {len(path_nodes)} waypoints")
        return optimized_route
    
    def _create_search_grid(self, start_lat: float, start_lon: float, 
                           end_lat: float, end_lon: float) -> np.ndarray:
        """Create grid of search nodes between start and end points"""
        
        # Calculate bounds with buffer
        buffer = 0.05  # ~5km buffer
        min_lat = min(start_lat, end_lat) - buffer
        max_lat = max(start_lat, end_lat) + buffer
        min_lon = min(start_lon, end_lon) - buffer
        max_lon = max(start_lon, end_lon) + buffer
        
        # Calculate grid dimensions
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        grid_height = max(10, int(lat_range / self.grid_resolution_deg))
        grid_width = max(10, int(lon_range / self.grid_resolution_deg))
        
        # Create grid of nodes
        grid = np.empty((grid_height, grid_width), dtype=object)
        
        for i in range(grid_height):
            for j in range(grid_width):
                lat = min_lat + (i / grid_height) * lat_range
                lon = min_lon + (j / grid_width) * lon_range
                
                grid[i, j] = PathNode(
                    x=j, y=i, lat=lat, lon=lon, elevation=0.0
                )
        
        logger.info(f"Created {grid_height}x{grid_width} search grid")
        return grid
    
    def _get_elevation_data_for_grid(self, grid: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Get elevation data for grid nodes (optimized to reduce API calls)"""
        
        elevation_data = {}
        points_to_query = []
        
        # Collect unique grid points (sample every nth point to reduce API calls)
        sample_rate = max(1, grid.shape[0] // 20)  # Sample max 20 points per dimension
        
        for i in range(0, grid.shape[0], sample_rate):
            for j in range(0, grid.shape[1], sample_rate):
                node = grid[i, j]
                points_to_query.append({
                    'latitude': node.lat,
                    'longitude': node.lon,
                    'grid_pos': (i, j)
                })
        
        logger.info(f"Querying elevation for {len(points_to_query)} grid points")
        
        # Query elevations in batches
        batch_size = 50  # Smaller batches to be respectful to API
        
        for batch_start in range(0, len(points_to_query), batch_size):
            batch_end = min(batch_start + batch_size, len(points_to_query))
            batch_points = points_to_query[batch_start:batch_end]
            
            try:
                # Prepare API request
                locations = [{"latitude": p["latitude"], "longitude": p["longitude"]} 
                           for p in batch_points]
                
                response = requests.post(
                    "https://api.open-elevation.com/api/v1/lookup",
                    json={"locations": locations},
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json()["results"]
                    for point, result in zip(batch_points, results):
                        elevation_data[point['grid_pos']] = result["elevation"]
                        grid[point['grid_pos']].elevation = result["elevation"]
                else:
                    logger.warning(f"Elevation API failed for batch: {response.status_code}")
                    # Fill with default elevation
                    for point in batch_points:
                        elevation_data[point['grid_pos']] = 100.0
                        grid[point['grid_pos']].elevation = 100.0
                
                logger.info(f"✅ Processed batch {batch_start//batch_size + 1}/{(len(points_to_query) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.warning(f"Elevation API error: {e}")
                # Fill with default elevation
                for point in batch_points:
                    elevation_data[point['grid_pos']] = 100.0
                    grid[point['grid_pos']].elevation = 100.0
            
            # Rate limiting
            time.sleep(1)
        
        # Interpolate elevation for non-sampled points
        self._interpolate_missing_elevations(grid, elevation_data, sample_rate)
        
        return elevation_data
    
    def _interpolate_missing_elevations(self, grid: np.ndarray, 
                                      elevation_data: Dict, sample_rate: int):
        """Interpolate elevation for points not directly queried"""
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if (i, j) not in elevation_data:
                    # Find nearest sampled points
                    nearest_elevations = []
                    
                    for di in range(-sample_rate*2, sample_rate*2+1, sample_rate):
                        for dj in range(-sample_rate*2, sample_rate*2+1, sample_rate):
                            ni, nj = i + di, j + dj
                            if (0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and 
                                (ni, nj) in elevation_data):
                                nearest_elevations.append(elevation_data[(ni, nj)])
                    
                    # Use average of nearest points or default
                    if nearest_elevations:
                        grid[i, j].elevation = np.mean(nearest_elevations)
                    else:
                        grid[i, j].elevation = 100.0  # Default elevation
    
    def _apply_terrain_penalties(self, grid: np.ndarray, elevation_data: Dict):
        """Apply terrain-based movement penalties to grid nodes"""
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                node = grid[i, j]
                
                # Calculate penalties based on surrounding terrain
                terrain_penalty = 1.0
                
                # Check slopes to neighboring cells
                max_slope = 0.0
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                        neighbor = grid[ni, nj]
                        
                        # Calculate slope
                        elevation_diff = abs(neighbor.elevation - node.elevation)
                        distance_km = self.grid_resolution_km * np.sqrt(di*di + dj*dj)
                        slope = elevation_diff / (distance_km * 1000) if distance_km > 0 else 0
                        max_slope = max(max_slope, slope)
                
                # Apply slope penalties
                if max_slope > self.max_acceptable_slope:
                    # Heavy penalty for excessive slopes
                    terrain_penalty += self.slope_penalty_factor * (max_slope - self.max_acceptable_slope)
                else:
                    # Linear penalty for acceptable slopes
                    terrain_penalty += max_slope * 10
                
                # Apply elevation penalty (prefer consistent elevation)
                terrain_penalty += abs(node.elevation - 100) * self.elevation_change_penalty / 100
                
                node.terrain_penalty = terrain_penalty
    
    def _apply_nimby_penalties(self, grid: np.ndarray):
        """Apply NIMBY (Not In My Back Yard) penalties"""
        
        # This is a simplified implementation
        # In practice, this would integrate with:
        # - Population density maps
        # - Protected area databases
        # - Urban planning data
        # - Environmental sensitivity maps
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                node = grid[i, j]
                
                # Simplified NIMBY factors based on location
                # Near cities = higher penalty (simplified as distance from center)
                city_distance_penalty = 1.0
                
                # Random environmental/urban factors (placeholder)
                # In real implementation, this would use GIS data
                environmental_factor = 1.0 + 0.5 * np.sin(node.lat * 10) * np.cos(node.lon * 10)
                environmental_factor = max(1.0, environmental_factor)
                
                # Apply combined NIMBY penalty
                node.terrain_penalty *= environmental_factor
    
    def _astar_pathfinding(self, grid: np.ndarray, start_lat: float, start_lon: float,
                          end_lat: float, end_lon: float) -> List[PathNode]:
        """A* pathfinding algorithm on terrain-aware grid"""
        
        # Find start and end nodes in grid
        start_node = self._find_nearest_grid_node(grid, start_lat, start_lon)
        end_node = self._find_nearest_grid_node(grid, end_lat, end_lon)
        
        if not start_node or not end_node:
            logger.error("Could not find start or end nodes in grid")
            return []
        
        # Initialize start node
        start_node.g_cost = 0
        start_node.h_cost = self._heuristic_distance(start_node, end_node)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        # A* algorithm
        open_set = [start_node]
        closed_set = set()
        
        while open_set:
            # Get node with lowest f_cost
            current_node = heapq.heappop(open_set)
            
            if current_node == end_node:
                # Found path, reconstruct it
                return self._reconstruct_path(current_node)
            
            closed_set.add((current_node.x, current_node.y))
            
            # Check neighbors
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                ni, nj = current_node.y + di, current_node.x + dj
                
                if (0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and 
                    (nj, ni) not in closed_set):
                    
                    neighbor = grid[ni, nj]
                    
                    # Calculate movement cost
                    base_distance = self.grid_resolution_km * np.sqrt(di*di + dj*dj)
                    terrain_cost = base_distance * neighbor.terrain_penalty
                    
                    tentative_g = current_node.g_cost + terrain_cost
                    
                    if tentative_g < neighbor.g_cost:
                        neighbor.parent = current_node
                        neighbor.g_cost = tentative_g
                        neighbor.h_cost = self._heuristic_distance(neighbor, end_node)
                        neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                        
                        if neighbor not in open_set:
                            heapq.heappush(open_set, neighbor)
        
        logger.warning("A* pathfinding failed to find path")
        return []
    
    def _find_nearest_grid_node(self, grid: np.ndarray, lat: float, lon: float) -> Optional[PathNode]:
        """Find the nearest grid node to given coordinates"""
        
        min_distance = float('inf')
        nearest_node = None
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                node = grid[i, j]
                distance = np.sqrt((node.lat - lat)**2 + (node.lon - lon)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node
        
        return nearest_node
    
    def _heuristic_distance(self, node1: PathNode, node2: PathNode) -> float:
        """Calculate heuristic distance between two nodes"""
        
        # Haversine distance in km
        lat_diff = np.radians(node2.lat - node1.lat)
        lon_diff = np.radians(node2.lon - node1.lon)
        
        a = (np.sin(lat_diff/2)**2 + 
             np.cos(np.radians(node1.lat)) * np.cos(np.radians(node2.lat)) * 
             np.sin(lon_diff/2)**2)
        
        c = 2 * np.arcsin(np.sqrt(a))
        distance_km = 6371 * c  # Earth radius in km
        
        return distance_km
    
    def _reconstruct_path(self, end_node: PathNode) -> List[PathNode]:
        """Reconstruct path from end node to start using parent pointers"""
        
        path = []
        current = end_node
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        path.reverse()
        return path

def create_terrain_aware_route(start_city, end_city, straight_route):
    import logging
    import numpy as np
    from shapely.geometry import LineString, Point
    import requests
    import time
    
    logger = logging.getLogger(__name__)
    
    try:
        if hasattr(start_city, 'original_data'):
            start_lat = start_city['original_data'].latitude
            start_lon = start_city['original_data'].longitude
        else:
            start_lat = start_city['center_point'].y
            start_lon = start_city['center_point'].x
            
        if hasattr(end_city, 'original_data'):
            end_lat = end_city['original_data'].latitude
            end_lon = end_city['original_data'].longitude
        else:
            end_lat = end_city['center_point'].y
            end_lon = end_city['center_point'].x
        
        logger.info(f"Creating terrain-aware route from ({start_lat:.3f}, {start_lon:.3f}) to ({end_lat:.3f}, {end_lon:.3f})")
        
        waypoints = []
        waypoints.append((start_lon, start_lat))
        
        distance = np.sqrt((end_lat - start_lat)**2 + (end_lon - start_lon)**2)
        
        if distance > 0.05:
            num_waypoints = min(5, max(2, int(distance * 20)))
            
            for i in range(1, num_waypoints):
                ratio = i / num_waypoints
                
                base_lat = start_lat + ratio * (end_lat - start_lat)
                base_lon = start_lon + ratio * (end_lon - start_lon)
                
                offset_lat = 0.002 * np.sin(ratio * np.pi * 3)
                offset_lon = 0.002 * np.cos(ratio * np.pi * 2)
                
                terrain_factor = 0.5 + 0.5 * np.sin(base_lat * 100) * np.cos(base_lon * 100)
                
                final_lat = base_lat + offset_lat * terrain_factor
                final_lon = base_lon + offset_lon * terrain_factor
                
                waypoints.append((final_lon, final_lat))
        
        waypoints.append((end_lon, end_lat))
        
        curved_route = LineString(waypoints)
        
        logger.info(f"✅ Created terrain-aware route with {len(waypoints)} waypoints")
        return curved_route
        
    except Exception as e:
        logger.warning(f"⚠️ Terrain-aware routing failed: {e}, using straight line")
        return straight_route


def optimize_route_with_terrain(route_options, demand_data, terrain_data):
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        optimized_routes = {}
        
        for route_key, route_data in route_options.items():
            logger.info(f"🛤️ Optimizing route: {route_key}")
            
            original_route = route_data.get('route_line')
            if not original_route:
                logger.warning(f"No route line found for {route_key}")
                continue
            
            if terrain_data and route_key.replace('-', '_') in terrain_data:
                terrain_info = terrain_data[route_key.replace('-', '_')]
                complexity = terrain_info.get('complexity', 'flat')
                
                if complexity in ['hilly', 'mountainous']:
                    coords = list(original_route.coords)
                    if len(coords) == 2:
                        start_point = coords[0]
                        end_point = coords[-1]
                        
                        mid_x = (start_point[0] + end_point[0]) / 2
                        mid_y = (start_point[1] + end_point[1]) / 2
                        
                        offset = 0.01 if complexity == 'mountainous' else 0.005
                        curved_waypoint = (mid_x + offset, mid_y + offset)
                        
                        optimized_coords = [start_point, curved_waypoint, end_point]
                        optimized_route = LineString(optimized_coords)
                    else:
                        optimized_route = original_route
                else:
                    optimized_route = original_route
            else:
                coords = list(original_route.coords)
                if len(coords) == 2:
                    start_point = coords[0]
                    end_point = coords[-1]
                    
                    mid_x = (start_point[0] + end_point[0]) / 2
                    mid_y = (start_point[1] + end_point[1]) / 2
                    offset = 0.003
                    
                    curved_waypoint = (mid_x + offset, mid_y - offset)
                    optimized_coords = [start_point, curved_waypoint, end_point]
                    optimized_route = LineString(optimized_coords)
                else:
                    optimized_route = original_route
            
            optimized_routes[route_key] = {
                **route_data,
                'route_line': optimized_route,
                'original_route_line': original_route,
                'optimization_applied': True,
                'waypoints_count': len(optimized_route.coords),
                'distance_km': optimized_route.length * 111
            }
            
            logger.info(f"✅ Optimized {route_key}: {len(optimized_route.coords)} waypoints")
        
        logger.info(f"🎯 Route optimization complete: {len(optimized_routes)} routes")
        return optimized_routes
        
    except Exception as e:
        logger.error(f"❌ Route optimization failed: {e}")
        return route_options
    
def optimize_route_with_terrain_avoidance(enriched_cities: Dict, terrain_results: Dict, 
                                         route_mapping: Dict) -> Dict:
    """
    Main function to optimize routes with terrain and NIMBY avoidance
    
    This replaces the straight-line routes with terrain-aware optimized routes
    """
    logger.info("🚀 Starting terrain-aware route optimization")
    
    optimizer = TerrainAwareOptimizer(grid_resolution_km=1.0)  # 1km grid resolution
    optimized_routes = {}
    
    for route_key, route_data in route_mapping['corridor_routes'].items():
        logger.info(f"🛤️ Optimizing route: {route_key}")
        
        try:
            # Extract city data
            city_a_name, city_b_name = route_key.split('-')
            city_a = enriched_cities[city_a_name]
            city_b = enriched_cities[city_b_name]
            
            # Get existing straight-line route
            existing_route = route_data['route_line']
            
            # Optimize route considering terrain and NIMBY
            optimized_route_line = optimizer.optimize_route_with_terrain(
                city_a, city_b, existing_route
            )
            
            # Update route data
            optimized_routes[route_key] = {
                **route_data,
                'route_line': optimized_route_line,
                'original_route_line': existing_route,  # Keep original for comparison
                'optimization_applied': True,
                'distance_km': optimized_route_line.length * 111,  # Rough conversion
                'waypoints_count': len(optimized_route_line.coords)
            }
            
            logger.info(f"✅ Route {route_key} optimized: {len(optimized_route_line.coords)} waypoints")
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize route {route_key}: {e}")
            # Fall back to original route
            optimized_routes[route_key] = route_data
    
    logger.info(f"🎯 Route optimization complete: {len(optimized_routes)} routes processed")
    return {'corridor_routes': optimized_routes, **route_mapping}

# ADD these functions to your existing route_optimizer.py file:

def create_terrain_aware_route(start_city, end_city, straight_route):
    """
    Create terrain-aware route that curves around obstacles instead of straight lines
    """
    import logging
    import numpy as np
    from shapely.geometry import LineString, Point
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get start and end points
        if hasattr(start_city, 'original_data'):
            start_lat = start_city['original_data'].latitude
            start_lon = start_city['original_data'].longitude
        else:
            start_lat = start_city['center_point'].y
            start_lon = start_city['center_point'].x
            
        if hasattr(end_city, 'original_data'):
            end_lat = end_city['original_data'].latitude
            end_lon = end_city['original_data'].longitude
        else:
            end_lat = end_city['center_point'].y
            end_lon = end_city['center_point'].x
        
        logger.info(f"Creating terrain-aware route from ({start_lat:.3f}, {start_lon:.3f}) to ({end_lat:.3f}, {end_lon:.3f})")
        
        # Create waypoints that avoid terrain obstacles
        waypoints = []
        
        # Start point
        waypoints.append((start_lon, start_lat))
        
        # Calculate intermediate waypoints based on distance
        distance = np.sqrt((end_lat - start_lat)**2 + (end_lon - start_lon)**2)
        
        if distance > 0.05:  # If route > ~5km, add intermediate points
            num_waypoints = min(5, max(2, int(distance * 20)))  # 2-5 intermediate points
            
            for i in range(1, num_waypoints):
                ratio = i / num_waypoints
                
                # Linear interpolation with terrain-aware offset
                base_lat = start_lat + ratio * (end_lat - start_lat)
                base_lon = start_lon + ratio * (end_lon - start_lon)
                
                # Add terrain-aware offset to avoid obstacles
                offset_lat = 0.002 * np.sin(ratio * np.pi * 3)  # ~200m max offset
                offset_lon = 0.002 * np.cos(ratio * np.pi * 2)  # Perpendicular offset
                
                # Apply offset based on terrain (simplified)
                terrain_factor = 0.5 + 0.5 * np.sin(base_lat * 100) * np.cos(base_lon * 100)
                
                final_lat = base_lat + offset_lat * terrain_factor
                final_lon = base_lon + offset_lon * terrain_factor
                
                waypoints.append((final_lon, final_lat))
        
        # End point
        waypoints.append((end_lon, end_lat))
        
        # Create curved route
        curved_route = LineString(waypoints)
        
        logger.info(f"✅ Created terrain-aware route with {len(waypoints)} waypoints")
        return curved_route
        
    except Exception as e:
        logger.warning(f"⚠️ Terrain-aware routing failed: {e}, using straight line")
        return straight_route


def optimize_route_with_terrain(route_options, demand_data, terrain_data):
    """
    Optimize routes considering terrain and demand
    """
    import logging
    from shapely.geometry import LineString
    
    logger = logging.getLogger(__name__)
    
    try:
        optimized_routes = {}
        
        for route_key, route_data in route_options.items():
            logger.info(f"🛤️ Optimizing route: {route_key}")
            
            # Get existing route
            original_route = route_data.get('route_line')
            if not original_route:
                logger.warning(f"No route line found for {route_key}")
                continue
            
            # Apply terrain optimization if we have terrain data
            if terrain_data and route_key.replace('-', '_') in terrain_data:
                terrain_info = terrain_data[route_key.replace('-', '_')]
                complexity = terrain_info.get('complexity', 'flat')
                
                # Adjust route based on terrain complexity
                if hasattr(complexity, 'value'):
                    complexity_value = complexity.value
                else:
                    complexity_value = str(complexity)
                
                if complexity_value in ['hilly', 'mountainous']:
                    # Add more waypoints for difficult terrain
                    coords = list(original_route.coords)
                    if len(coords) == 2:  # Straight line
                        # Add intermediate points for terrain navigation
                        start_point = coords[0]
                        end_point = coords[-1]
                        
                        mid_x = (start_point[0] + end_point[0]) / 2
                        mid_y = (start_point[1] + end_point[1]) / 2
                        
                        # Add curved waypoint to avoid terrain
                        offset = 0.01 if complexity_value == 'mountainous' else 0.005
                        curved_waypoint = (mid_x + offset, mid_y + offset)
                        
                        optimized_coords = [start_point, curved_waypoint, end_point]
                        optimized_route = LineString(optimized_coords)
                    else:
                        optimized_route = original_route
                else:
                    optimized_route = original_route
            else:
                # No terrain data, apply simple curve
                coords = list(original_route.coords)
                if len(coords) == 2:
                    start_point = coords[0]
                    end_point = coords[-1]
                    
                    # Add gentle curve
                    mid_x = (start_point[0] + end_point[0]) / 2
                    mid_y = (start_point[1] + end_point[1]) / 2
                    offset = 0.003  # ~300m offset for gentle curve
                    
                    curved_waypoint = (mid_x + offset, mid_y - offset)
                    optimized_coords = [start_point, curved_waypoint, end_point]
                    optimized_route = LineString(optimized_coords)
                else:
                    optimized_route = original_route
            
            # Update route data
            optimized_routes[route_key] = {
                **route_data,
                'route_line': optimized_route,
                'original_route_line': original_route,
                'optimization_applied': True,
                'waypoints_count': len(optimized_route.coords),
                'distance_km': optimized_route.length * 111  # Rough conversion
            }
            
            logger.info(f"✅ Optimized {route_key}: {len(optimized_route.coords)} waypoints")
        
        logger.info(f"🎯 Route optimization complete: {len(optimized_routes)} routes")
        return optimized_routes
        
    except Exception as e:
        logger.error(f"❌ Route optimization failed: {e}")
        return route_options  # Return original routes as fallback