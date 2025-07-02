"""
Load the single master CSV, split it to canonical DataFrames, validate
lat/lon coverage, and write pretty-printed JSON files to data/.
"""

import os
import pathlib as pl
import pandas as pd
import json, warnings, glob

# ───────────────────────────────────────────────────────────────
#  0. Work out which CSV to load
# ───────────────────────────────────────────────────────────────
ROOT       = pl.Path(__file__).resolve().parents[1]
INPUT_DIR  = ROOT / "input"

# Priority: (1) env var  (2) exactly-one *.csv in input/  (3) master-data.csv
if os.getenv("MASTER_CSV"):
    INPUT = pl.Path(os.getenv("MASTER_CSV")).expanduser()
else:
    csv_files = list(INPUT_DIR.glob("*.csv"))
    if len(csv_files) == 1:
        INPUT = csv_files[0]
    else:
        INPUT = INPUT_DIR / "master-data.csv"

COSTS  = ROOT / "input" / "cost_params.json"
OUTPUT = ROOT / "data"

# ───────────────────────────────────────────────────────────────
#  C. Cost parameters (optional override)
# ───────────────────────────────────────────────────────────────
DEFAULT_COSTS = {
    "track_cost_per_km": {"coastal": 7.5, "rolling": 9.0, "mountain": 18.0},
    "double_track_multiplier": 1.6,
    "high_speed_multiplier": 2.1,
    "train_cost": {"TER_4car": 15.0, "TGV_8car": 38.0},
    "annual_crew_cost_per_train": 0.45,
    "discount_rate": 0.05
}

def load_cost_params() -> dict:
    if COSTS.exists():
        with open(COSTS) as f:
            return DEFAULT_COSTS | json.load(f)
    warnings.warn("cost_params.json not found – using baked-in defaults")
    return DEFAULT_COSTS

# ───────────────────────────────────────────────────────────────
#  A. Parse CSV → canonical DataFrames
# ───────────────────────────────────────────────────────────────
def parse_master_csv(path: pl.Path | None = None) -> dict[str, pd.DataFrame]:
    path = path or INPUT
    if not path.exists():
        raise FileNotFoundError(f"CSV input not found: {path}")

    df   = pd.read_csv(path)
    data: dict[str, pd.DataFrame] = {}

    # node-level
    data["Stations"] = (
        df[["station_id", "city", "name", "n_tracks",
            "size_m2", "amenities", "overhead_wires"]]
        .dropna(subset=["station_id"])
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )

    data["Population-per-city"] = (
        df[["city", "population"]]
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    data["City-coords"] = (
        df[["city", "lat", "lon"]]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    # edge-level
    edge_rows = df[df["segment_id"].notna()]

    data["Tracks"] = (
        edge_rows[["segment_id", "city_a", "city_b", "distance_km",
                   "terrain_class", "allowed_train_types"]]
        .drop_duplicates("segment_id")
        .reset_index(drop=True)
    )

    data["Frequency"] = (
        edge_rows[["segment_id", "peak_trains_per_hour",
                   "offpeak_trains_per_hour"]]
        .drop_duplicates("segment_id")
        .reset_index(drop=True)
    )

    # project-wide
    data["Budget"] = (
        df[["fiscal_year", "currency", "capex_million", "opex_million"]]
        .drop_duplicates("fiscal_year")
        .reset_index(drop=True)
    )

    # ── validation: ensure every city on a track has lat/lon -------------
    track_cities = set(data["Tracks"]["city_a"]) | set(data["Tracks"]["city_b"])
    track_cities = {c for c in track_cities if isinstance(c, str) and c.strip()}
    coord_cities = set(data["City-coords"]["city"])
    missing = track_cities - coord_cities

    if missing:
        raise ValueError(
            f"🚫 Missing lat/lon for: {', '.join(sorted(missing))}. "
            "Add these rows to your CSV."
        )


    return data

# ───────────────────────────────────────────────────────────────
#  JSON writer
# ───────────────────────────────────────────────────────────────
def materialise_json(dfs: dict[str, pd.DataFrame], out_dir: pl.Path = OUTPUT):
    out_dir.mkdir(exist_ok=True, parents=True)
    for name, frame in dfs.items():
        (out_dir / f"{name}.json").write_text(
            frame.to_json(orient="records", indent=2)
        )

# ───────────────────────────────────────────────────────────────
#  Back-compat helper
# ───────────────────────────────────────────────────────────────
def read_json(name: str) -> pd.DataFrame:
    path = OUTPUT / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found – run parse_master_csv()/materialise_json() first."
        )
    return pd.read_json(path)
