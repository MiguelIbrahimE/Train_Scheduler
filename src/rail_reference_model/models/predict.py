"""
Embed *new* rail graphs with a previously trained Node2Vec model,
and (optionally) assign them to existing cluster centroids.

Usage
-----
python -m rail_reference_model.models.predict \
       model_dir=node2vec_be \
       graphml=new_corridor.graphml \
       out_embeddings=new_corridor_emb.npy
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
import typer
from sklearn.cluster import KMeans

from .gnn_encoder import RailNode2Vec

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Infer embeddings & clusters for new graphs")


def _load_centroids(model_dir: Path) -> KMeans | None:
    km_file = model_dir / "kmeans.joblib"
    if km_file.exists():
        import joblib

        return joblib.load(km_file)
    return None


@app.command()
def run(
    model_dir: Path = typer.Argument(..., help="Folder containing node2vec.pt"),
    graphml: Path = typer.Argument(..., help="GraphML to embed"),
    out_embeddings: Path = typer.Option(..., help="Destination .npy for embeddings"),
    out_json: Path = typer.Option(None, help="Optional JSON with node→cluster"),
) -> None:
    # ------------------------------------------------------------------ #
    # load model & target graph
    # ------------------------------------------------------------------ #
    model = RailNode2Vec.load(model_dir / "node2vec.pt")
    G = nx.read_graphml(graphml)
    G = nx.convert_node_labels_to_integers(G)
    model.fit(G, epochs=0)  # rebuild internal structures without training

    emb = model.embed_nodes().numpy()
    out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_embeddings, emb)
    logger.info("Embeddings saved to %s", out_embeddings)

    if out_json:
        km: KMeans | None = _load_centroids(model_dir)
        if km is None:
            logger.warning("No kmeans.joblib in %s — clustering skipped", model_dir)
            return

        labels = km.predict(emb)
        mapping: Dict[str, Any] = {
            "node_id": list(range(len(labels))),
            "cluster": labels.tolist(),
        }
        with open(out_json, "w", encoding="utf8") as fh:
            json.dump(mapping, fh, indent=2)
        logger.info("Node-cluster map saved → %s", out_json)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    app()
