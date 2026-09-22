"""Fast lookups for routing actions and local observation windows."""

from __future__ import annotations

import math

import networkx as nx


class GraphWrapper:
    def __init__(self, graph: "nx.MultiDiGraph", max_degree: int | None = None):
        self.graph = graph
        if max_degree is None:
            max_degree = max((len(list(graph.successors(node))) for node in graph.nodes), default=0)
        if max_degree < 0:
            raise ValueError(f"max_degree must be non-negative, got {max_degree!r}")
        self.max_degree = int(max_degree)
        self._actions_by_node: dict = {}
        for node in graph.nodes:
            outgoing = []
            for u, v, k, data in graph.out_edges(node, keys=True, data=True):
                outgoing.append({
                    "edge": (u, v, k),
                    "bearing": self._bearing_between(node, v),
                    "data": data,
                })
            outgoing.sort(key=lambda item: item["bearing"])
            padded = [None] * self.max_degree
            for idx, item in enumerate(outgoing[: self.max_degree]):
                padded[idx] = item["edge"]
            self._actions_by_node[node] = padded

    def _node_coords(self, node):
        node_data = self.graph.nodes[node]
        x = node_data.get("x")
        y = node_data.get("y")
        if x is None or y is None:
            raise ValueError(f"node {node!r} is missing x/y coordinates")
        return float(x), float(y)

    def _bearing_between(self, src_node, dst_node):
        lon1, lat1 = self._node_coords(src_node)
        lon2, lat2 = self._node_coords(dst_node)

        dlon = math.radians(lon2 - lon1)
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        y = math.sin(dlon) * math.cos(lat2_r)
        x = (
            math.cos(lat1_r) * math.sin(lat2_r)
            - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        )
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360.0) % 360.0

    def action_to_edge(self, node, action_idx: int):
        """Return (u, v, k) for the edge at action index *node* or None if invalid."""
        if node not in self.graph:
            return None
        if not isinstance(action_idx, int) or action_idx < 0 or action_idx >= self.max_degree:
            return None
        return self._actions_by_node.get(node, [None] * self.max_degree)[action_idx]

    def action_mask(self, node) -> list[bool]:
        """Boolean mask for valid actions at this node."""
        if node not in self.graph:
            return [False] * self.max_degree
        actions = self._actions_by_node.get(node, [None] * self.max_degree)
        return [edge is not None for edge in actions]

    def neighbors_within_hops(self, node, hops: int) -> list:
        """Return the edges within a local reveal radius of the current node."""
        if hops < 0:
            return []
        if node not in self.graph:
            return []
        if hops == 0:
            return []
        undirected = self.graph.to_undirected(as_view=True)
        distances = nx.single_source_shortest_path_length(undirected, node, cutoff=hops)
        edges = []
        for u, v, k in self.graph.edges(keys=True):
            if u in distances or v in distances:
                edges.append((u, v, k))
        return edges
