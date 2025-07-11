"""
report.py – Simple HTML/XLSX export (placeholder).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimise.optimise import NetworkDesign


def export_summary(design: NetworkDesign, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "metric": ["total_cost_eur", "ridership_daily"],
            "value": [design.cost_eur, design.ridership_daily],
        }
    )
    out_file = out_dir / "summary.xlsx"
    df.to_excel(out_file, index=False)
    return out_file
