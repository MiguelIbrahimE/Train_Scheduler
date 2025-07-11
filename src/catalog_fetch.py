from __future__ import annotations
import logging, time, json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup   # html.parser; stdlib would also work

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_TRACK_FILE  = CACHE_DIR / "track_catalog.yml"
_TRAIN_FILE  = CACHE_DIR / "train_catalog.yml"
_REFRESH_AGE = timedelta(days=7)

log = logging.getLogger("bcpc.catalog")


# ───────────────────────────────────────────────
def _needs_refresh(path: Path) -> bool:
    return not path.exists() or (
        datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
    ) > _REFRESH_AGE


# ───────────────────────────────────────────────
def _scrape_track_catalog() -> pd.DataFrame:
    """Return DF with columns: name, gauge_mm, electrified, vmax_kmh, min_rad_m."""
    url = "https://en.wikipedia.org/wiki/List_of_high-speed_rail_lines"
    tables = pd.read_html(url, flavor="bs4")
    df = tables[0]  # first table on the page
    df = df.rename(
        columns={
            "Line": "name",
            "Design speed (km/h)": "vmax_kmh",
            "Gauge (mm)": "gauge_mm",
            "Minimum curve radius (m)": "min_rad_m",
        }
    )
    df["electrified"] = True   # all high-speed lines in that list are
    df = df[["name", "gauge_mm", "electrified", "vmax_kmh", "min_rad_m"]]
    return df


# ───────────────────────────────────────────────
def _scrape_train_catalog() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_high-speed_trains"
    tables = pd.read_html(url, flavor="bs4")
    df = tables[0]
    df = df.rename(
        columns={
            "Train name": "name",
            "Top speed\n(km/h)": "top_speed_kmh",
            "Gauge\n(mm)": "gauge_mm",
            "Capacity": "capacity",
            "Cost (per set)": "purchase_cost_eur",
        }
    )
    # clean € strings → float
    df["purchase_cost_eur"] = (
        df["purchase_cost_eur"]
        .str.replace(r"[^\d.]", "", regex=True)
        .astype(float)
        * 1_000_000  # most rows show “€30 million” etc.
    )
    df = df[["name", "gauge_mm", "capacity", "top_speed_kmh", "purchase_cost_eur"]]
    return df


# ───────────────────────────────────────────────
def _write_yaml(df: pd.DataFrame, path: Path):
    path.write_text(df.to_yaml(index=False, allow_nan=False))


# ───────────────────────────────────────────────
def fetch_catalogs(force: bool = False):
    """Fetch remote catalogs and cache to YAML.  Returns (track_df, train_df)."""
    track_df = train_df = None

    try:
        if force or _needs_refresh(_TRACK_FILE):
            track_df = _scrape_track_catalog()
            _write_yaml(track_df, _TRACK_FILE)
            log.info("Track catalog refreshed from Wikipedia")
        if force or _needs_refresh(_TRAIN_FILE):
            train_df = _scrape_train_catalog()
            _write_yaml(train_df, _TRAIN_FILE)
            log.info("Train catalog refreshed from Wikipedia")
    except Exception as exc:  # noqa: BLE001
        log.warning("Live catalog fetch failed (%s) – falling back to cache", exc)

    # load cache (guaranteed to exist after first successful run)
    if track_df is None:
        track_df = pd.read_yaml(_TRACK_FILE)
    if train_df is None:
        train_df = pd.read_yaml(_TRAIN_FILE)

    return track_df, train_df
