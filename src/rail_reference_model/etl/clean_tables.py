"""
ETL ‒ KPI table cleaner
-----------------------

Parses rail-operator annual-report PDFs or CSVs into a canonical schema:

    country | year | kpi                 | value | unit

Supported KPIs (extend as needed):
    - punctuality_5m   (%)
    - pass_km_per_km   (pkm / route-km)
    - capex_track_km   (M€2025 / track-km)

Usage example::

    python -m rail_reference_model.etl.clean_tables input/raw/ns_2024.pdf \
           data/processed/kpis_ns.parquet
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import pdfplumber
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Clean KPI tables into tidy format")

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")


def _extract_numbers(text: str) -> float | None:
    """Return first numeric value in a string or None."""
    match = _NUMBER_RE.search(text.replace(",", "."))
    return float(match.group()) if match else None


def _parse_pdf(pdf_path: Path) -> pd.DataFrame:
    """
    Very simple PDF table parser using pdfplumber text search.

    Assumes a page contains lines like:
        Punctuality (<5 min)       89.3 %
        CAPEX / track-km           1.7   M€

    Extend with Camelot/Tabula if PDFs are more structured.
    """
    rows: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "Punctuality" in text:
                pct = _extract_numbers(text.split("Punctuality")[1])
                rows.append(
                    dict(kpi="punctuality_5m", value=pct, unit="percent")
                )
            if "CAPEX" in text and "track" in text:
                capex = _extract_numbers(text.split("CAPEX")[1])
                rows.append(
                    dict(kpi="capex_track_km", value=capex, unit="MEUR_2025_per_km")
                )
            # add additional pattern blocks here…

    if not rows:
        logger.warning("No KPI rows parsed from %s", pdf_path)
    return pd.DataFrame(rows)


def _tidy(df: pd.DataFrame, country: str, year: int) -> pd.DataFrame:
    """Attach metadata columns and canonical order."""
    df["country"] = country.lower()
    df["year"] = year
    return df[["country", "year", "kpi", "value", "unit"]]


@app.command()
def run(
    infile: Path = typer.Argument(..., help="Raw PDF or CSV"),
    outfile: Path = typer.Argument(..., help="Destination Parquet/CSV file"),
    country: str = typer.Option(..., "--country", "-c", help="ISO country code"),
    year: int = typer.Option(..., "--year", "-y", help="Report year (e.g. 2024)"),
) -> None:
    """
    Clean *infile* into a tidy KPI table saved to *outfile*.
    """
    if infile.suffix.lower() == ".pdf":
        df = _parse_pdf(infile)
    else:
        raise typer.BadParameter("Only PDF parsing implemented for now")

    df = _tidy(df, country, year)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(outfile, index=False)
    else:
        df.to_csv(outfile, index=False)

    logger.info("Wrote %s (%d rows)", outfile, len(df))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s ‒ %(message)s"
    )
    app()
