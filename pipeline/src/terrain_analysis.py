"""
BCPC Pipeline - Terrain Analysis Module
======================================

This module provides comprehensive terrain analysis for railway route planning,
including elevation profiles, slope calculations, terrain complexity assessment,
and integration with OpenElevation API for global DEM data.

Features:
- OpenElevation API integration (free, no API key required)
- Terrain complexity classification for cost analysis
- Slope and curvature analysis for engineering constraints
- Tunnel and bridge requirement identification
- Integration with station placement and cost analysis modules
- Caching for offline deterministic reruns

Author: BCPC Pipeline Team
License: Open Source
"""

import json
import logging
import math
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import transform
import pyproj
from scipy import ndimage
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, time
_OE_LOCK = threading.BoundedSemaphore(2)      # never hammer the service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DEMSource(Enum):
    """Available DEM data sources"""
    OPENELEVATION = "OpenElevation"  # Free API, no key required
    FLAT = "Flat"                    # Fallback flat terrain

class TerrainComplexity(Enum):
    """Terrain complexity classification for cost analysis"""
    FLAT = "flat"                 # 0-2% grade, minimal earthwork
    ROLLING = "rolling"           # 2-4% grade, moderate earthwork
    HILLY = "hilly"              # 4-6% grade, significant earthwork
    MOUNTAINOUS = "mountainous"   # 6%+ grade, extensive tunnels/bridges
    URBAN = "urban"              # Flat but complex due to existing infrastructure

class RailwayConstraints:
    """Railway engineering constraints"""
    MAX_GRADE_PASSENGER = 0.025    # 2.5% maximum grade for passenger trains
    MAX_GRADE_FREIGHT = 0.015      # 1.5% maximum grade for freight trains
    MIN_CURVE_RADIUS = 1000        # 1000m minimum curve radius for high-speed
    MAX_TUNNEL_GRADE = 0.015       # 1.5% maximum grade in tunnels
    BRIDGE_THRESHOLD_HEIGHT = 30   # 30m minimum height for major bridges
    TUNNEL_THRESHOLD_HEIGHT = 50   # 50m minimum height for tunnel consideration

@dataclass
class ElevationProfile:
    """Elevation profile along a route"""
    distances: np.ndarray          # Distance along route (km)
    elevations: np.ndarray         # Elevation values (m)
    slopes: np.ndarray             # Slope percentages
    curvatures: np.ndarray         # Horizontal curvature (1/radius)
    total_length_km: float
    min_elevation: float
    max_elevation: float
    elevation_gain: float
    elevation_loss: float

@dataclass
class TerrainSegment:
    """Individual terrain segment with characteristics"""
    start_km: float
    end_km: float
    complexity: TerrainComplexity
    avg_slope: float
    max_slope: float
    elevation_change: float
    earthwork_volume_estimate: float
    requires_tunnel: bool = False
    requires_bridge: bool = False
    tunnel_length_km: float = 0.0
    bridge_length_km: float = 0.0

@dataclass
class TerrainAnalysis:
    """Complete terrain analysis results"""
    route_line: LineString
    elevation_profile: ElevationProfile
    terrain_segments: List[TerrainSegment]
    overall_complexity: TerrainComplexity
    total_earthwork_volume: float
    total_tunnel_length_km: float
    total_bridge_length_km: float
    cost_multiplier: float          # Relative to flat terrain
    construction_feasibility: float # 0-1 score
    dem_source: DEMSource
    resolution_meters: float

@dataclass
class StationTerrainContext:
    """Terrain context for station placement"""
    station_location: Point
    local_slope: float
    elevation: float
    platform_earthwork_required: bool
    access_road_difficulty: float  # 0-1 score
    drainage_challenges: bool
    construction_cost_factor: float

class TerrainAnalyzer:
    """
    Main terrain analysis engine for BCPC railway projects using OpenElevation API
    """
    
    def __init__(self, 
                 cache_dir: str = "data/_cache/terrain",
                 preferred_resolution: float = 250.0):
        """
        Initialize terrain analyzer
        
        Args:
            cache_dir: Directory for caching DEM data
            preferred_resolution: Preferred DEM resolution in meters (250m for OpenElevation)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preferred_resolution = preferred_resolution
        
        # Analysis parameters
        self.profile_sample_interval = 100  # meters between profile samples
        self.segment_length_km = 5.0        # km length for terrain segments
        self.slope_smoothing_window = 5     # number of points for slope smoothing
        
    def analyze_route_terrain(self, 
                            route_line: LineString,
                            buffer_km: float = 2.0,
                            dem_source: Optional[DEMSource] = None) -> TerrainAnalysis:
        """
        Perform comprehensive terrain analysis for a railway route
        
        Args:
            route_line: Railway route geometry
            buffer_km: Buffer around route for DEM download
            dem_source: Preferred DEM source, auto-selected if None
            
        Returns:
            Complete terrain analysis results
        """
        logger.info(f"Analyzing terrain for {route_line.length/1000:.1f}km route")
        
        # Download and cache DEM data
        dem_array, dem_transform, actual_source = self._get_route_dem(
            route_line, buffer_km, dem_source
        )
        
        # Extract elevation profile along route
        elevation_profile = self._extract_elevation_profile(route_line, dem_array, dem_transform)
        
        # Analyze terrain segments
        terrain_segments = self._analyze_terrain_segments(elevation_profile)
        
        # Calculate overall metrics
        overall_complexity = self._determine_overall_complexity(terrain_segments)
        total_earthwork = sum(seg.earthwork_volume_estimate for seg in terrain_segments)
        total_tunnel = sum(seg.tunnel_length_km for seg in terrain_segments)
        total_bridge = sum(seg.bridge_length_km for seg in terrain_segments)
        
        # Calculate cost multiplier and feasibility
        cost_multiplier = self._calculate_cost_multiplier(terrain_segments, overall_complexity)
        feasibility = self._assess_construction_feasibility(elevation_profile, terrain_segments)
        
        return TerrainAnalysis(
            route_line=route_line,
            elevation_profile=elevation_profile,
            terrain_segments=terrain_segments,
            overall_complexity=overall_complexity,
            total_earthwork_volume=total_earthwork,
            total_tunnel_length_km=total_tunnel,
            total_bridge_length_km=total_bridge,
            cost_multiplier=cost_multiplier,
            construction_feasibility=feasibility,
            dem_source=actual_source,
            resolution_meters=self.preferred_resolution
        )
    
    def analyze_station_terrain(self, 
                              station_locations: List[Point],
                              route_line: LineString,
                              terrain_analysis: TerrainAnalysis) -> List[StationTerrainContext]:
        """
        Analyze terrain context for station locations
        
        Args:
            station_locations: List of proposed station locations
            route_line: Railway route line
            terrain_analysis: Previously computed terrain analysis
            
        Returns:
            Terrain context for each station
        """
        logger.info(f"Analyzing terrain context for {len(station_locations)} stations")
        
        station_contexts = []
        
        for i, station_point in enumerate(station_locations):
            # Find position along route
            route_position = route_line.project(station_point)
            route_distance_km = route_position / 1000
            
            # Interpolate elevation and slope at station location
            elevation = np.interp(
                route_distance_km,
                terrain_analysis.elevation_profile.distances,
                terrain_analysis.elevation_profile.elevations
            )
            
            local_slope = np.interp(
                route_distance_km,
                terrain_analysis.elevation_profile.distances[:-1],
                terrain_analysis.elevation_profile.slopes
            )
            
            # Assess station-specific challenges
            platform_earthwork = abs(local_slope) > 0.005  # 0.5% slope requires earthwork
            access_difficulty = min(1.0, abs(local_slope) * 20)  # Scale slope to 0-1
            drainage_challenges = elevation < 100 or local_slope < -0.02  # Low elevation or steep downgrade
            
            # Calculate construction cost factor
            cost_factor = 1.0 + abs(local_slope) * 10 + (0.5 if platform_earthwork else 0)
            cost_factor = min(3.0, cost_factor)  # Cap at 3x base cost
            
            station_contexts.append(StationTerrainContext(
                station_location=station_point,
                local_slope=local_slope,
                elevation=elevation,
                platform_earthwork_required=platform_earthwork,
                access_road_difficulty=access_difficulty,
                drainage_challenges=drainage_challenges,
                construction_cost_factor=cost_factor
            ))
        
        return station_contexts
    
    def _get_route_dem(self, 
                      route_line: LineString,
                      buffer_km: float,
                      dem_source: Optional[DEMSource]) -> Tuple[np.ndarray, rasterio.transform.Affine, DEMSource]:
        """Download and cache DEM data for route area"""
        
        # Create bounding box with buffer
        bounds = route_line.bounds
        buffer_deg = buffer_km / 111.0  # Rough conversion km to degrees
        
        west = bounds[0] - buffer_deg
        south = bounds[1] - buffer_deg
        east = bounds[2] + buffer_deg
        north = bounds[3] + buffer_deg
        
        # Generate cache filename
        cache_filename = f"dem_{west:.4f}_{south:.4f}_{east:.4f}_{north:.4f}.tif"
        cache_path = self.cache_dir / cache_filename
        
        # Check cache first
        if cache_path.exists():
            logger.info(f"Loading cached DEM from {cache_path}")
            with rasterio.open(cache_path) as src:
                dem_array = src.read(1)
                dem_transform = src.transform
                actual_source = DEMSource.OPENELEVATION  # Default assumption for cached data
                return dem_array, dem_transform, actual_source
        
        # Try OpenElevation API
        try:
            logger.info("Downloading elevation data from OpenElevation API")
            dem_array, dem_transform = self._download_from_openelevation(west, south, east, north)
            self._cache_dem(dem_array, dem_transform, cache_path, west, south, east, north)
            logger.info("✅ Successfully downloaded DEM from OpenElevation")
            return dem_array, dem_transform, DEMSource.OPENELEVATION
        except Exception as e:
            logger.warning(f"❌ OpenElevation API failed: {e}")
            logger.warning("⚠️  Falling back to flat terrain assumption")
            return self._create_flat_dem(west, south, east, north)
    
    def _download_from_openelevation(self, west: float, south: float, east: float, north: float) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """Download elevation data from OpenElevation API"""
        import time
        
        # Create grid of points (coarser resolution for API limits)
        step = 0.0025  # ~250m resolution
        lats = np.arange(south, north + step, step)
        lons = np.arange(west, east + step, step)
        
        # Create elevation grid
        elevation_grid = np.zeros((len(lats), len(lons)))
        
        # Query points in batches to avoid API limits
        batch_size = 100  # Reasonable batch size
        points = []
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                points.append({
                    "latitude": lat,
                    "longitude": lon,
                    "grid_i": i,
                    "grid_j": j
                })
        
        logger.info(f"Querying {len(points)} elevation points from OpenElevation API")
        
        # Process in batches
        for batch_start in range(0, len(points), batch_size):
            batch_end = min(batch_start + batch_size, len(points))
            batch_points = points[batch_start:batch_end]
            
            # Prepare API request
            locations = [{"latitude": p["latitude"], "longitude": p["longitude"]} 
                        for p in batch_points]
            
            try:
                response = requests.post(
                    "https://api.open-elevation.com/api/v1/lookup",
                    json={"locations": locations},
                    timeout=60
                )
                
                if response.status_code == 200:
                    results = response.json()["results"]
                    for point, result in zip(batch_points, results):
                        elevation_grid[point["grid_i"], point["grid_j"]] = result["elevation"]
                    logger.info(f"✅ Processed batch {batch_start//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
                else:
                    logger.warning(f"❌ OpenElevation API batch failed: {response.status_code}")
                    # Fill with default elevation
                    for point in batch_points:
                        elevation_grid[point["grid_i"], point["grid_j"]] = 100.0
            
            except Exception as e:
                logger.warning(f"❌ OpenElevation API error: {e}")
                # Fill with default elevation
                for point in batch_points:
                    elevation_grid[point["grid_i"], point["grid_j"]] = 100.0
            
            # Rate limiting - be respectful to free API
            time.sleep(2)
        
        # Create transform
        transform = rasterio.transform.from_bounds(west, south, east, north, len(lons), len(lats))
        
        return elevation_grid, transform
    
    def _cache_dem(self, 
                  dem_array: np.ndarray,
                  dem_transform: rasterio.transform.Affine,
                  cache_path: Path,
                  west: float, south: float, east: float, north: float) -> None:
        """Cache DEM data to local file"""
        
        height, width = dem_array.shape
        
        with rasterio.open(
            cache_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=dem_array.dtype,
            crs='EPSG:4326',
            transform=dem_transform,
            compress='lzw'
        ) as dst:
            dst.write(dem_array, 1)
        
        logger.info(f"Cached DEM to {cache_path}")
    
    def _create_flat_dem(self, 
                        west: float, south: float, 
                        east: float, north: float) -> Tuple[np.ndarray, rasterio.transform.Affine, DEMSource]:
        """Create a flat DEM as fallback"""
        
        # Create grid with reasonable resolution
        resolution = 0.0025  # ~250m in degrees
        width = int((east - west) / resolution)
        height = int((north - south) / resolution)
        
        # Create flat terrain at 100m elevation
        dem_array = np.full((height, width), 100.0, dtype=np.float32)
        
        # Create transform
        dem_transform = from_bounds(west, south, east, north, width, height)
        
        return dem_array, dem_transform, DEMSource.FLAT
    
    def _extract_elevation_profile(self,
                                 route_line: LineString,
                                 dem_array: np.ndarray,
                                 dem_transform: rasterio.transform.Affine) -> ElevationProfile:
        """Extract elevation profile along the route"""
        
        # Sample points along route
        route_length = route_line.length
        num_samples = max(100, int(route_length / self.profile_sample_interval))
        
        distances = np.linspace(0, route_length, num_samples)
        sample_points = [route_line.interpolate(d) for d in distances]
        
        # Extract elevations
        elevations = []
        for point in sample_points:
            row, col = rasterio.transform.rowcol(dem_transform, point.x, point.y)
            
            # Handle bounds checking
            if (0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]):
                elevation = dem_array[row, col]
                if np.isnan(elevation):
                    elevation = 100.0  # Default elevation for nodata
            else:
                elevation = 100.0  # Default for out-of-bounds
            
            elevations.append(elevation)
        
        elevations = np.array(elevations)
        distances_km = distances / 1000  # Convert to km
        
        # Calculate slopes
        slopes = np.diff(elevations) / np.diff(distances)  # m/m
        slopes = np.append(slopes, slopes[-1])  # Extend to match length
        
        # Smooth slopes
        if len(slopes) > self.slope_smoothing_window:
            slopes = ndimage.uniform_filter1d(slopes, size=self.slope_smoothing_window)
        
        # Calculate curvatures (simplified)
        curvatures = np.zeros_like(slopes)
        if len(slopes) > 2:
            curvatures[1:-1] = np.diff(slopes, 2)
        
        # Calculate profile statistics
        elevation_gain = np.sum(np.diff(elevations)[np.diff(elevations) > 0])
        elevation_loss = abs(np.sum(np.diff(elevations)[np.diff(elevations) < 0]))
        
        return ElevationProfile(
            distances=distances_km,
            elevations=elevations,
            slopes=slopes,
            curvatures=curvatures,
            total_length_km=route_length / 1000,
            min_elevation=np.min(elevations),
            max_elevation=np.max(elevations),
            elevation_gain=elevation_gain,
            elevation_loss=elevation_loss
        )
    
    def _analyze_terrain_segments(self, elevation_profile: ElevationProfile) -> List[TerrainSegment]:
        """Analyze terrain in segments along the route"""
        
        segments = []
        total_length = elevation_profile.total_length_km
        segment_starts = np.arange(0, total_length, self.segment_length_km)
        
        for i, start_km in enumerate(segment_starts):
            end_km = min(start_km + self.segment_length_km, total_length)
            
            # Find indices for this segment
            start_idx = np.argmin(np.abs(elevation_profile.distances - start_km))
            end_idx = np.argmin(np.abs(elevation_profile.distances - end_km))
            
            if start_idx == end_idx:
                continue
            
            # Extract segment data
            seg_elevations = elevation_profile.elevations[start_idx:end_idx+1]
            seg_slopes = elevation_profile.slopes[start_idx:end_idx]
            
            # Calculate segment characteristics
            avg_slope = np.mean(np.abs(seg_slopes))
            max_slope = np.max(np.abs(seg_slopes))
            elevation_change = seg_elevations[-1] - seg_elevations[0]
            
            # Determine complexity
            complexity = self._classify_terrain_complexity(avg_slope, max_slope)
            
            # Estimate earthwork volume (simplified)
            segment_length_m = (end_km - start_km) * 1000
            earthwork_volume = self._estimate_earthwork_volume(
                seg_slopes, segment_length_m, complexity
            )
            
            # Check for tunnel/bridge requirements
            requires_tunnel, tunnel_length = self._assess_tunnel_requirement(
                seg_slopes, seg_elevations, segment_length_m
            )
            
            requires_bridge, bridge_length = self._assess_bridge_requirement(
                seg_slopes, seg_elevations, segment_length_m
            )
            
            segments.append(TerrainSegment(
                start_km=start_km,
                end_km=end_km,
                complexity=complexity,
                avg_slope=avg_slope,
                max_slope=max_slope,
                elevation_change=elevation_change,
                earthwork_volume_estimate=earthwork_volume,
                requires_tunnel=requires_tunnel,
                requires_bridge=requires_bridge,
                tunnel_length_km=tunnel_length / 1000,
                bridge_length_km=bridge_length / 1000
            ))
        
        return segments
    
    def _classify_terrain_complexity(self, avg_slope: float, max_slope: float) -> TerrainComplexity:
        """Classify terrain complexity based on slopes"""
        
        if max_slope > 0.06:  # 6%
            return TerrainComplexity.MOUNTAINOUS
        elif max_slope > 0.04:  # 4%
            return TerrainComplexity.HILLY
        elif avg_slope > 0.02:  # 2%
            return TerrainComplexity.ROLLING
        else:
            return TerrainComplexity.FLAT
    
    def _estimate_earthwork_volume(self,
                                 slopes: np.ndarray,
                                 segment_length_m: float,
                                 complexity: TerrainComplexity) -> float:
        """Estimate earthwork volume for a segment"""
        
        # Simplified earthwork estimation
        avg_slope = np.mean(np.abs(slopes))
        
        # Base earthwork per meter of route
        if complexity == TerrainComplexity.FLAT:
            base_volume_per_m = 50    # m³/m of route
        elif complexity == TerrainComplexity.ROLLING:
            base_volume_per_m = 200
        elif complexity == TerrainComplexity.HILLY:
            base_volume_per_m = 500
        else:  # MOUNTAINOUS
            base_volume_per_m = 1000
        
        # Adjust for actual slopes
        slope_multiplier = 1 + avg_slope * 20
        
        total_volume = base_volume_per_m * segment_length_m * slope_multiplier
        
        return total_volume
    
    def _assess_tunnel_requirement(self,
                                 slopes: np.ndarray,
                                 elevations: np.ndarray,
                                 segment_length_m: float) -> Tuple[bool, float]:
        """Assess if tunnels are required in a segment"""
        
        max_slope = np.max(np.abs(slopes))
        elevation_range = np.max(elevations) - np.min(elevations)
        
        # Tunnel required if slopes exceed railway limits and significant elevation change
        requires_tunnel = (max_slope > RailwayConstraints.MAX_GRADE_PASSENGER and 
                          elevation_range > RailwayConstraints.TUNNEL_THRESHOLD_HEIGHT)
        
        if requires_tunnel:
            # Estimate tunnel length as portion of segment with excessive slopes
            excessive_slope_indices = np.where(np.abs(slopes) > RailwayConstraints.MAX_GRADE_PASSENGER)[0]
            if len(excessive_slope_indices) > 0:
                tunnel_length = len(excessive_slope_indices) / len(slopes) * segment_length_m
            else:
                tunnel_length = segment_length_m * 0.3  # Default 30% of segment
        else:
            tunnel_length = 0.0
        
        return requires_tunnel, tunnel_length
    
    def _assess_bridge_requirement(self,
                                 slopes: np.ndarray,
                                 elevations: np.ndarray,
                                 segment_length_m: float) -> Tuple[bool, float]:
        """Assess if bridges are required in a segment"""
        
        # Simple bridge assessment - look for significant elevation changes
        elevation_range = np.max(elevations) - np.min(elevations)
        
        # Bridge required for valley crossings or to maintain grades
        requires_bridge = elevation_range > RailwayConstraints.BRIDGE_THRESHOLD_HEIGHT
        
        if requires_bridge:
            # Estimate bridge length based on elevation change and slope requirements
            max_grade = RailwayConstraints.MAX_GRADE_PASSENGER
            required_length = elevation_range / max_grade
            bridge_length = min(required_length, segment_length_m * 0.5)  # Max 50% of segment
        else:
            bridge_length = 0.0
        
        return requires_bridge, bridge_length
    
    def _determine_overall_complexity(self, segments: List[TerrainSegment]) -> TerrainComplexity:
        """Determine overall terrain complexity for the route"""
        
        complexity_weights = {
            TerrainComplexity.FLAT: 1,
            TerrainComplexity.ROLLING: 2,
            TerrainComplexity.HILLY: 3,
            TerrainComplexity.MOUNTAINOUS: 4
        }
        
        # Weight by segment length
        total_length = sum(seg.end_km - seg.start_km for seg in segments)
        weighted_complexity = 0
        
        for segment in segments:
            segment_length = segment.end_km - segment.start_km
            weight = segment_length / total_length
            weighted_complexity += complexity_weights[segment.complexity] * weight
        
        # Map back to complexity class
        if weighted_complexity >= 3.5:
            return TerrainComplexity.MOUNTAINOUS
        elif weighted_complexity >= 2.5:
            return TerrainComplexity.HILLY
        elif weighted_complexity >= 1.5:
            return TerrainComplexity.ROLLING
        else:
            return TerrainComplexity.FLAT
    
    def _calculate_cost_multiplier(self,
                                 segments: List[TerrainSegment],
                                 overall_complexity: TerrainComplexity) -> float:
        """Calculate cost multiplier relative to flat terrain"""
        
        base_multipliers = {
            TerrainComplexity.FLAT: 1.0,
            TerrainComplexity.ROLLING: 1.6,
            TerrainComplexity.HILLY: 2.5,
            TerrainComplexity.MOUNTAINOUS: 4.0
        }
        
        base_multiplier = base_multipliers[overall_complexity]
        
        # Additional costs for tunnels and bridges
        total_length = sum(seg.end_km - seg.start_km for seg in segments)
        total_tunnel = sum(seg.tunnel_length_km for seg in segments)
        total_bridge = sum(seg.bridge_length_km for seg in segments)
        
        if total_length > 0:
            tunnel_factor = (total_tunnel / total_length) * 10  # Tunnels are ~10x more expensive
            bridge_factor = (total_bridge / total_length) * 5   # Bridges are ~5x more expensive
        else:
            tunnel_factor = bridge_factor = 0
        
        final_multiplier = base_multiplier + tunnel_factor + bridge_factor
        
        return min(final_multiplier, 8.0)  # Cap at 8x base cost
    
    def _assess_construction_feasibility(self,
                                       profile: ElevationProfile,
                                       segments: List[TerrainSegment]) -> float:
        """Assess overall construction feasibility (0-1 score)"""
        
        # Start with base feasibility
        feasibility = 1.0
        
        # Penalize for excessive slopes
        max_slope = np.max(np.abs(profile.slopes))
        if max_slope > RailwayConstraints.MAX_GRADE_PASSENGER * 2:
            feasibility -= 0.3
        elif max_slope > RailwayConstraints.MAX_GRADE_PASSENGER:
            feasibility -= 0.1
        
        # Penalize for excessive elevation changes
        elevation_range = profile.max_elevation - profile.min_elevation
        if elevation_range > 1000:  # 1000m
            feasibility -= 0.2
        elif elevation_range > 500:  # 500m
            feasibility -= 0.1
        
        # Penalize for high tunnel/bridge requirements
        total_length = profile.total_length_km
        tunnel_ratio = sum(seg.tunnel_length_km for seg in segments) / total_length
        bridge_ratio = sum(seg.bridge_length_km for seg in segments) / total_length
        
        if tunnel_ratio > 0.3:  # >30% tunnels
            feasibility -= 0.3
        elif tunnel_ratio > 0.1:  # >10% tunnels
            feasibility -= 0.1
        
        if bridge_ratio > 0.2:  # >20% bridges
            feasibility -= 0.2
        elif bridge_ratio > 0.05:  # >5% bridges
            feasibility -= 0.1
        
        return max(0.1, feasibility)  # Minimum 0.1 feasibility

def analyze_terrain_lightweight(route_line, city_name="Unknown"):
    import logging
    import requests
    import numpy as np
    import threading
    import time
    from shapely.geometry import LineString
    
    logger = logging.getLogger(__name__)
    _api_lock = threading.Lock()
    
    try:
        with _api_lock:
            route_length = route_line.length
            num_points = min(15, max(5, int(route_length * 500)))
            
            sample_points = []
            for i in range(num_points):
                distance_ratio = i / (num_points - 1) if num_points > 1 else 0
                point = route_line.interpolate(distance_ratio, normalized=True)
                sample_points.append(point)
            
            locations = [{"latitude": p.y, "longitude": p.x} for p in sample_points]
            
            logger.info(f"Querying {len(locations)} elevation points for {city_name}")
            
            response = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": locations},
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()["results"]
                elevations = [r["elevation"] for r in results]
                
                logger.info(f"✅ Got elevation data for {city_name}")
                
                if len(elevations) > 1:
                    elevation_range = max(elevations) - min(elevations)
                    max_slope = 0.0
                    
                    for i in range(len(elevations) - 1):
                        height_diff = abs(elevations[i+1] - elevations[i])
                        distance_km = route_line.length * 111 / len(elevations)
                        if distance_km > 0:
                            slope = height_diff / (distance_km * 1000)
                            max_slope = max(max_slope, slope)
                    
                    if max_slope > 0.06:
                        complexity = TerrainComplexity.MOUNTAINOUS
                        cost_multiplier = 4.0
                    elif max_slope > 0.04:
                        complexity = TerrainComplexity.HILLY 
                        cost_multiplier = 2.5
                    elif max_slope > 0.02:
                        complexity = TerrainComplexity.ROLLING
                        cost_multiplier = 1.6
                    else:
                        complexity = TerrainComplexity.FLAT
                        cost_multiplier = 1.0
                        
                    feasibility = max(0.3, 1.0 - (max_slope * 10))
                    
                else:
                    complexity = TerrainComplexity.FLAT
                    cost_multiplier = 1.0
                    feasibility = 0.8
                    elevation_range = 0
                    
                result = type('TerrainAnalysis', (), {
                    'overall_complexity': complexity,
                    'cost_multiplier': cost_multiplier,
                    'construction_feasibility': feasibility,
                    'total_tunnel_length_km': elevation_range / 1000 * 0.1 if elevation_range > 500 else 0,
                    'total_bridge_length_km': elevation_range / 1000 * 0.05 if elevation_range > 200 else 0,
                    'total_earthwork_volume': elevation_range * 1000,
                    'route_line': route_line,
                    'elevation_profile': type('ElevationProfile', (), {
                        'elevations': np.array(elevations),
                        'distances': np.linspace(0, route_line.length * 111, len(elevations)),
                        'total_length_km': route_line.length * 111
                    })()
                })()
                
                logger.info(f"✅ Terrain analysis complete for {city_name}: {complexity.value}")
                return result
                
            else:
                logger.warning(f"⚠️ Elevation API returned status {response.status_code} for {city_name}")
                
    except Exception as e:
        logger.warning(f"❌ Terrain analysis failed for {city_name}: {e}")
    
    logger.info(f"Using flat terrain fallback for {city_name}")
    result = type('TerrainAnalysis', (), {
        'overall_complexity': TerrainComplexity.FLAT,
        'cost_multiplier': 1.0,
        'construction_feasibility': 0.8,
        'total_tunnel_length_km': 0,
        'total_bridge_length_km': 0,
        'total_earthwork_volume': 0,
        'route_line': route_line,
        'elevation_profile': type('ElevationProfile', (), {
            'elevations': np.array([100.0, 100.0]),
            'distances': np.array([0, route_line.length * 111]),
            'total_length_km': route_line.length * 111
        })()
    })()
    
    return result

def create_mock_terrain_analysis(route_line):
    """Create mock terrain analysis for testing without API calls"""
    
    route_length_km = route_line.length * 111
    
    if route_length_km < 100:
        complexity = TerrainComplexity.ROLLING
        cost_multiplier = 1.6
        feasibility = 0.8
    else:
        complexity = TerrainComplexity.HILLY
        cost_multiplier = 2.5
        feasibility = 0.6
    
    result = type('TerrainAnalysis', (), {
        'overall_complexity': complexity,
        'cost_multiplier': cost_multiplier,
        'construction_feasibility': feasibility,
        'total_tunnel_length_km': route_length_km * 0.1,
        'total_bridge_length_km': route_length_km * 0.05,
        'total_earthwork_volume': route_length_km * 1000,
        'route_line': route_line,
        'elevation_profile': type('ElevationProfile', (), {
            'elevations': np.array([100.0, 150.0, 120.0]),
            'distances': np.array([0, route_length_km/2, route_length_km]),
            'total_length_km': route_length_km
        })()
    })()
    
    return result