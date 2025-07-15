# ---- base: Python 3.11 with Torch 2.2 CPU wheels ----
    FROM python:3.11-bullseye AS base

    ENV DEBIAN_FRONTEND=noninteractive \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    # System libs for GDAL + PyG source builds (lightweight)
    RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential cmake ninja-build libgdal-dev gdal-bin wget ca-certificates \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # ---- Python deps ----------------------------------------------------
    COPY pyproject.toml poetry.lock ./
    RUN pip install --upgrade pip \
     && pip install poetry==1.8.2 \
     && poetry config virtualenvs.create false \
     && poetry install --no-interaction --without dev
    
    # Torch-Geometric wheels that match Torch 2.2.0 CPU
    RUN pip install \
          torch_scatter torch_sparse torch_cluster pyg_lib \
          --no-index -f https://data.pyg.org/whl/torch-2.2.0+cpu.html \
     && pip install torch_geometric==2.6.1
    
    # ---- copy source ----------------------------------------------------
    COPY src ./src
    ENV PYTHONPATH=/app/src
    
    # ---- default entry = FastAPI ----------------------------------------
    EXPOSE 8000
    CMD ["uvicorn", "rail_reference_model.inference.service:app", "--host", "0.0.0.0", "--port", "8000"]
    