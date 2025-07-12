"""
Simple K-means (default) or HDBSCAN clustering on node embeddings.

Example
-------
python -m rail_reference_model.models.cluster \
       model_dir=node2vec_be \
       k=6 \
       out_file=node2vec_be/clusters.npy
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import typer
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Cluster Node2Vec embeddings")


@app.command()
def run(
    model_dir: Path = typer.Argument(..., help="Folder with node_embeddings.npy"),
    k: int = typer.Option(8, help="Number of clusters (K-means)"),
    out_file: Path = typer.Option(
        None, help="Optional path to save cluster labels (.npy)"
    ),
) -> None:
    emb = np.load(model_dir / "node_embeddings.npy")
    logger.info("Clustering %s (shape %s) into k=%d …", model_dir, emb.shape, k)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(emb)
    logger.info("Cluster counts: %s", np.bincount(labels))

    if out_file:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_file, labels)
        logger.info("Labels saved → %s", out_file)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    app()
