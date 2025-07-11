import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
from enum import Enum        # ← add this
import folium
from shapely.geometry import LineString, Point, mapping
from shapely import wkt

from utils.geometry import densify
from utils.terrain import classify_segment, InfrastructureType
from utils.terrain import TerrainComplexity  # reuse if defined there, else keep enum below

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainType(Enum):
    ELECTRIC_EMU = "electric_emu"


class StationType(Enum):
    INTERMEDIATE = "intermediate"


@dataclass
class RouteSegmentVisualization:
    geometry: LineString
    infrastructure_type: InfrastructureType
    terrain_complexity: TerrainComplexity
    distance_km: float


@dataclass
class StationVisualization:
    location: Point
    name: str
    station_type: StationType


@dataclass
class RouteVisualization:
    route_id: str
    route_name: str
    segments: List[RouteSegmentVisualization]
    stations: List[StationVisualization]
    train_type: TrainType
    total_length_km: float
    total_cost: float
    overall_terrain: TerrainComplexity
    daily_passengers_total: int


class BCPCVisualizer:
    def create_route_from_bcpc_data(
        self,
        route_line: LineString | str,
        dem_path: str,
        terrain_analysis: Any = None,
        station_network: Any = None,
        cost_summary: Any = None,
        network_design: Any = None,
        route_name: str = "Route",
    ) -> RouteVisualization:
        if isinstance(route_line, str):
            route_line = wkt.loads(route_line)
        route_line = densify(route_line, step_km=0.03)
        segments_raw = classify_segment(route_line, dem_path)
        segments = [
            RouteSegmentVisualization(
                geometry=s,
                infrastructure_type=tag,
                terrain_complexity=TerrainComplexity.FLAT,
                distance_km=s.length * 111,
            )
            for s, tag in segments_raw
        ]
        distance_km = sum(seg.distance_km for seg in segments)
        return RouteVisualization(
            route_id=route_name.lower().replace(" ", "_"),
            route_name=route_name,
            segments=segments,
            stations=[],
            train_type=TrainType.ELECTRIC_EMU,
            total_length_km=distance_km,
            total_cost=getattr(cost_summary, "total_capex", 0),
            overall_terrain=TerrainComplexity.FLAT,
            daily_passengers_total=0,
        )

    def create_comprehensive_visualization(
        self,
        routes: List[RouteVisualization],
        output_path: str = "main_map.html",
        title: str = "BCPC Map",
    ) -> None:
        if not routes:
            return
        first_pt = routes[0].segments[0].geometry.coords[0]
        m = folium.Map(location=[first_pt[1], first_pt[0]], zoom_start=8, tiles="OpenStreetMap")
        palette = {
            InfrastructureType.GROUND: "#1976D2",
            InfrastructureType.TUNNEL: "#C62828",
            InfrastructureType.VIADUCT: "#2E7D32",
        }
        for route in routes:
            for seg in route.segments:
                folium.GeoJson(
                    data=mapping(seg.geometry),
                    name=f"{route.route_name}-{seg.infrastructure_type.value}",
                    style_function=lambda _, c=palette[seg.infrastructure_type]: {"color": c, "weight": 4},
                    smooth_factor=1.3,
                ).add_to(m)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        logger.info(f"saved {output_path}")


def create_visualization_from_scenario(
    scenario_results: Dict[str, Any], dem_path: str, output_path: str = "main_map.html"
) -> None:
    vis = BCPCVisualizer()
    routes: List[RouteVisualization] = []
    for name, data in scenario_results.items():
        if isinstance(data, dict) and "route_line" in data:
            routes.append(
                vis.create_route_from_bcpc_data(data["route_line"], dem_path=dem_path, route_name=name)
            )
    vis.create_comprehensive_visualization(routes, output_path)


def create_complete_dashboard(
    scenario_results: Dict[str, Any], dem_path: str, output_dir: str = "dashboard"
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    create_visualization_from_scenario(scenario_results, dem_path, out / "main_map.html")
    (out / "index.html").write_text(
        "<html><body><h1>BCPC Dashboard</h1>"
        "<p><a href='main_map.html'>Interactive map</a></p></body></html>",
        encoding="utf-8",
    )


def create_dashboard_index(*args, **kwargs):
    pass
