"""
Node2Vec wrapper that learns unsupervised node embeddings for any
rail-network graph produced by `features.build_graph`.

The resulting model can be re-loaded to embed new graphs and to
feed downstream clustering / regression tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import Node2Vec

logger = logging.getLogger(__name__)


class RailNode2Vec:
    """
    Thin convenience wrapper around torch-geometric's Node2Vec.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 10,
        num_negative_samples: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        sparse: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        self._params = dict(
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=sparse,
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Node2Vec | None = None
        self.node_map: Dict[int, int] | None = None  # nx node id  -> idx in embedding matrix

    # ------------------------------------------------------------------ #
    # fitting & inference
    # ------------------------------------------------------------------ #
    def fit(self, G: nx.Graph, epochs: int = 30, batch_size: int = 1024) -> None:
        """
        Train Node2Vec on *G*.

        Parameters
        ----------
        G : nx.Graph
            Undirected graph with integer node ids.
        epochs : int
            SGD epochs.
        batch_size : int
            Batch size for walk sampling.
        """
        logger.info("Preparing data (%d nodes, %d edges) …", G.number_of_nodes(), G.number_of_edges())
        data = from_networkx(G).to(self.device)
        model = Node2Vec(
            data.edge_index,
            num_nodes=data.num_nodes,
            **self._params,
        ).to(self.device)

        loader = model.loader(batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

        logger.info("Training Node2Vec (%d epochs) …", epochs)
        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += float(loss)

            if epoch % 5 == 0 or epoch == epochs:
                logger.info("Epoch %d/%d – loss %.4f", epoch, epochs, total_loss / len(loader))

        self.model = model
        # node ids are already 0…N-1 if G came from networkx→PyG conversion
        self.node_map = {int(k): int(k) for k in G.nodes}

    def embed_nodes(self) -> torch.Tensor:
        """Return a tensor [N, embedding_dim] with node embeddings."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call 'fit' first.")
        return self.model().cpu()

    # ------------------------------------------------------------------ #
    # persistence helpers
    # ------------------------------------------------------------------ #
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model else None,
                "node_map": self.node_map,
                "params": self._params,
            },
            out_dir / "node2vec.pt",
        )
        logger.info("Model saved to %s", out_dir / "node2vec.pt")

    @classmethod
    def load(cls, ckpt_path: Path, device: str | torch.device | None = None) -> "RailNode2Vec":
        obj = cls(device=device)
        payload = torch.load(ckpt_path, map_location=obj.device)
        obj._params = payload["params"]
        obj.node_map = payload["node_map"]
        model = Node2Vec(
            torch.empty(2, 0, dtype=torch.long),  # dummy edge_index; will be refilled on embed
            num_nodes=len(obj.node_map),
            **obj._params,
        ).to(obj.device)
        model.load_state_dict(payload["state_dict"])
        obj.model = model
        logger.info("Loaded Node2Vec model from %s", ckpt_path)
        return obj
