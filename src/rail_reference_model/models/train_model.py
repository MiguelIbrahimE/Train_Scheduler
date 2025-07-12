"""
Command-line helper that

1. reads a GraphML rail network,
2. trains Node2Vec embeddings,
3. persists model + `.npy` embedding matrix for downstream tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import typer

from .gnn_encoder import RailNode2Vec

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Train Node2Vec rail embeddings")


@app.command()
def run(
    graphml: Path = typer.Argument(..., help="Input GraphML file"),
    out_dir: Path = typer.Argument(..., help="Output directory for model artefacts"),
    epochs: int = typer.Option(30, help="Training epochs"),
    dim: int = typer.Option(128, help="Embedding dimension"),
) -> None:
    G = nx.read_graphml(graphml)
    G = nx.convert_node_labels_to_integers(G)  # ensures 0…N-1

    model = RailNode2Vec(embedding_dim=dim)
    model.fit(G, epochs=epochs)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir)

    emb = model.embed_nodes().numpy()
    np.save(out_dir / "node_embeddings.npy", emb)
    logging.info("Embedding matrix saved (%s)", out_dir / "node_embeddings.npy")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    app()
