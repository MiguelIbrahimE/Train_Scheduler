"""
Quick script to check ScenarioRow attributes
Save as check_attributes.py and run: poetry run python check_attributes.py
"""
from pathlib import Path
from src.scenario_io import load_scenario

# Load the CSV
scenarios = load_scenario(Path("input/lebanon_cities_2024.csv"))

# Check first scenario
if scenarios:
    first = scenarios[0]
    print("ScenarioRow attributes:")
    print(dir(first))
    print("\nActual values:")
    for attr in dir(first):
        if not attr.startswith('_'):
            try:
                value = getattr(first, attr)
                print(f"{attr}: {value}")
            except:
                pass