from shapely.geometry import LineString

def densify(line: LineString, step_km: float = 0.03) -> LineString:
    """
    Return a new LineString with a vertex every `step_km` kilometres.
    step_km = 0.03 → ~30 m between points (≈2 000 points on a 70 km route).
    """
    total_km = line.length * 111          # lon/lat → km (good enough for short lines)
    n = max(int(total_km / step_km), 2)   # at least two points
    pts = [line.interpolate(i / n, normalized=True) for i in range(n + 1)]
    return LineString(pts)
