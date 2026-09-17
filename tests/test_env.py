"""
Validate the Gymnasium contract AND the project-specific invariants that
matter most here — a broken reward scale or leaked state can train
"successfully" while quietly invalidating the whole comparison against
baselines. Run these before spending GPU time on PPO.

TODO:
- test_reset_returns_valid_observation: shape/dtype matches observation_space
- test_step_returns_valid_observation
- test_episode_terminates: random-action rollout eventually terminates or
  truncates (no infinite loop)
- test_no_state_leakage: the observation at a node must NOT reveal the true
  condition of an edge the agent hasn't traversed yet — this is the whole
  partial-observability premise; a bug here silently defeats the project's
  research question
- test_reward_scale_sane: flood penalty should be meaningfully larger than
  typical travel-time differences, but not so large it swamps all learning
  signal from step 1 (log actual values, don't just guess)
- test_reproducibility: same seed -> same scenario realization (needed for
  fair baseline comparison in evaluation/)
"""

import pytest  # noqa: F401


def test_reset_returns_valid_observation():
    raise NotImplementedError("TODO")


def test_step_returns_valid_observation():
    raise NotImplementedError("TODO")


def test_episode_terminates():
    raise NotImplementedError("TODO")


def test_no_state_leakage():
    raise NotImplementedError("TODO")


def test_reward_scale_sane():
    raise NotImplementedError("TODO")


def test_reproducibility():
    raise NotImplementedError("TODO")
