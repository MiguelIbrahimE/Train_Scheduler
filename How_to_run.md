```markdown
# Train Scheduler - Quick Run Guide

## Old Pipeline (Raster):
```bash
# In your Train_Scheduler directory:
poetry run python -m src --csv input/lebanon_cities_2024.csv
```

```bash
poetry run python -m src --csv input/lebanon_cities_2024.csv --intercity
```


## New Pipeline (Data Structures Based + Raster):

```bash
python auto_run_pipeline.py
```

poetry run python pipeline/src/pipeline_initiator.py --csv input/lebanon_cities_2024.csv --output output --verbose

## ML Model