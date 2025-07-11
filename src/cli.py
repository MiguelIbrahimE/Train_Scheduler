from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from src import INPUT_DIR
from src.models import Gauge, TrackType, TrainType
from src.scenario_io import load_scenario
from src.city_pipeline import run_city_pipeline as run_pipeline
from src.intercity_planner import plan_intercity_network

STD_TRACK = TrackType("Std-Cat-160", Gauge.STANDARD, True, 160, 1_200, 12_000_000)
EMU = TrainType("4-car EMU", Gauge.STANDARD, 400, 160, 10_000_000, 8)

log = logging.getLogger("bcpc.cli")
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")


def _arrow_menu(options: Sequence[str]) -> int:
    import curses

    def _menu(stdscr):
        curses.curs_set(0)
        idx = 0
        while True:
            stdscr.clear()
            for i, opt in enumerate(options):
                mode = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addstr(i, 0, opt, mode)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(options)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(options)
            elif key in (curses.KEY_ENTER, 10, 13):
                return idx

    return curses.wrapper(_menu)


def _find_csv_files() -> list[Path]:
    return sorted(INPUT_DIR.glob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BCPC scenario pipeline")
    parser.add_argument("--csv", help="Path to scenario CSV")
    parser.add_argument("--intercity", action="store_true", help="Run intercity rail planner")
    parser.add_argument("--pop-threshold", type=int, default=100_000, help="Population cutoff for intercity")
    args = parser.parse_args()

    # -- locate CSV --------------------------------------------------------
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csvs = _find_csv_files()
        if not csvs:
            log.error("No CSV files found in %s", INPUT_DIR)
            sys.exit(1)
        if len(csvs) == 1:
            csv_path = csvs[0]
            log.info("Using %s", csv_path.name)
        else:
            csv_path = csvs[_arrow_menu([p.name for p in csvs])]

    log.info("CSV → %s", csv_path)

    # -- intercity planner -------------------------------------------------
    if args.intercity:
        output_path = Path("output/intercity_network.geojson")
        plan_intercity_network(str(csv_path), str(output_path), pop_threshold=args.pop_threshold)
        return

    # -- default city pipeline ---------------------------------------------
    rows = load_scenario(csv_path)
    for row in rows:
        run_pipeline(row, rows, STD_TRACK, EMU)
