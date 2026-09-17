"""
Baseline 1: shortest path using fixed (base/expected) edge weights — no
knowledge of the current episode's actual hazard realization.

This is the "naive" comparator: what a driver following a normal map app
with no live flood/traffic awareness would do.

TODO:
- Use networkx.shortest_path with weight="base_travel_time"
- Evaluate the resulting route against the HazardSimulator's true
  conditions for that episode (the route doesn't adapt if it hits a
  flooded edge — that's the point, it's the naive baseline)
"""

import networkx as nx  # noqa: F401


def static_shortest_path(graph, origin, destination):
    raise NotImplementedError("TODO")
