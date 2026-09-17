"""
TODO:
- test_static_path_is_valid: returned route is a valid path in the graph
  from origin to destination
- test_dynamic_path_uses_current_risk: changing flood_risk on an edge
  changes the dynamic baseline's chosen route (sanity check that it's
  actually using the risk-weighted cost, not falling back to raw length)
"""

import pytest  # noqa: F401


def test_static_path_is_valid():
    raise NotImplementedError("TODO")


def test_dynamic_path_uses_current_risk():
    raise NotImplementedError("TODO")
