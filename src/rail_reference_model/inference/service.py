"""
Minimal FastAPI micro-service that exposes a `/predict` endpoint.

POST a GraphML file → receive node embeddings (base64-encoded NumPy array)
and, if k-means centroids exist in the model directory, the cluster labels.

Run locally with::

    uvicorn rail_reference_model.inference.service:app --reload
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from ..models.gnn_encoder import RailNode2Vec
from ..models.predict import _load_centroids

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[2] / "model_artifacts"

app = FastAPI(
    title="Rail-Reference-Model Inference",
    version="0.1.0",
    description="Upload a GraphML rail corridor; receive Node2Vec embeddings "
    "and (optionally) cluster labels.",
)


def _embed_graph(graph_bytes: bytes) -> Dict[str, Any]:
    # read graph
    G = nx.read_graphml(io.BytesIO(graph_bytes))
    G = nx.convert_node_labels_to_integers(G)

    # load pre-trained model
    model = RailNode2Vec.load(MODEL_DIR / "node2vec.pt")
    model.fit(G, epochs=0)  # rebuild
    emb = model.embed_nodes().numpy()

    # encode array → base64 to stay JSON-serialisable
    emb_b64 = base64.b64encode(emb.tobytes()).decode()

    # optional clustering
    km = _load_centroids(MODEL_DIR)
    labels = km.predict(emb) if km else None

    return dict(
        embedding_dim=emb.shape[1],
        num_nodes=emb.shape[0],
        embeddings_b64=emb_b64,
        cluster_labels=labels.tolist() if labels is not None else None,
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Main inference endpoint."""
    try:
        payload = _embed_graph(await file.read())
        return JSONResponse(payload)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Prediction failed")
        return JSONResponse({"error": str(exc)}, status_code=500)
