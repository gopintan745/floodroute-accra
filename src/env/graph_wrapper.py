"""
Thin wrapper around the NetworkX graph exposing fast lookups the env needs
every step (neighbor lists, edge features, node feature vectors), so
accra_routing_env.py doesn't do raw NetworkX traversal in the hot loop.

TODO:
- Precompute adjacency lists and edge feature arrays at load time
- Provide `neighbors(node) -> list[edge_id]`,
  `edge_features(edge_id) -> np.ndarray`,
  `node_features(node) -> np.ndarray`
- Consider caching a fixed max-degree padding scheme here if you go with
  the invalid-action-masking approach from accra_routing_env.py
"""

import networkx as nx  # noqa: F401


class GraphWrapper:
    def __init__(self, graph: "nx.MultiDiGraph"):
        raise NotImplementedError("TODO")
