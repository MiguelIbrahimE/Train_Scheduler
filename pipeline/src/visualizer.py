import logging
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List
import folium
from shapely.geometry import LineString, Point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfrastructureType(Enum):
    GROUND = "ground"

class TrainType(Enum):
    ELECTRIC_EMU = "electric_emu"

class TerrainComplexity(Enum):
    FLAT = "flat"

class StationType(Enum):
    INTERMEDIATE = "intermediate"

@dataclass
class RouteSegmentVisualization:
    start_point: Point
    end_point: Point
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
    def create_route_from_bcpc_data(self,
                                    route_line: LineString,
                                    terrain_analysis: Any = None,
                                    station_network: Any = None,
                                    cost_summary: Any = None,
                                    network_design: Any = None,
                                    route_name: str = "Route") -> RouteVisualization:
        distance_km = route_line.length * 111
        segment = RouteSegmentVisualization(
            start_point=Point(route_line.coords[0]),
            end_point=Point(route_line.coords[-1]),
            infrastructure_type=InfrastructureType.GROUND,
            terrain_complexity=TerrainComplexity.FLAT,
            distance_km=distance_km
        )
        return RouteVisualization(
            route_id=route_name.lower().replace(" ", "_"),
            route_name=route_name,
            segments=[segment],
            stations=[],
            train_type=TrainType.ELECTRIC_EMU,
            total_length_km=distance_km,
            total_cost=getattr(cost_summary, "total_capex", 0),
            overall_terrain=TerrainComplexity.FLAT,
            daily_passengers_total=0
        )

    def create_comprehensive_visualization(self,
                                           routes: List[RouteVisualization],
                                           output_path: str = "main_map.html",
                                           title: str = "BCPC Map") -> None:
        if not routes:
            return
        first = routes[0].segments[0]
        m = folium.Map(
            location=[first.start_point.y, first.start_point.x],
            zoom_start=8,
            tiles="OpenStreetMap",
            attr="© OpenStreetMap"
        )
        for route in routes:
            for seg in route.segments:
                folium.PolyLine(
                    [[seg.start_point.y, seg.start_point.x],
                     [seg.end_point.y, seg.end_point.x]],
                    color="blue",
                    weight=4
                ).add_to(m)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        logger.info(f"saved {output_path}")

def create_visualization_from_scenario(scenario_results: Dict[str, Any],
                                       output_path: str = "main_map.html") -> None:
    vis = BCPCVisualizer()
    routes = []
    for name, data in scenario_results.items():
        if isinstance(data, dict) and "route_line" in data:
            routes.append(vis.create_route_from_bcpc_data(data["route_line"], route_name=name))
    vis.create_comprehensive_visualization(routes, output_path)

def create_complete_dashboard(scenario_results: Dict[str, Any],
                              output_dir: str = "dashboard") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    create_visualization_from_scenario(scenario_results, out / "main_map.html")
    (out / "index.html").write_text(
        "<html><body><h1>BCPC Dashboard</h1>"
        "<p><a href='main_map.html'>Interactive map</a></p></body></html>",
        encoding="utf-8"
    )

def create_dashboard_index(*args, **kwargs):
    pass
