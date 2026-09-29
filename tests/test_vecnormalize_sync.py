"""Test that VecNormalize statistics stay in sync between training and evaluation envs."""

import tempfile
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from src.agents.training_utils import load_config, load_graph, GRAPH_PATH, make_env_factory, make_normalized_vec_env
from src.env.accra_routing_env import AccraRoutingEnv


def test_vecnormalize_obs_rms_sync():
    """Training and evaluation VecNormalize should have identical obs_rms after sync."""
    config = load_config()
    graph = load_graph(GRAPH_PATH)

    # Create training vec_env
    train_vec_env = make_normalized_vec_env(graph, config, n_envs=2, mode="train")

    # Simulate some training steps to populate obs_rms
    for _ in range(10):
        train_vec_env.reset()
        train_vec_env.step([0, 0])

    # Save training stats to file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        train_vec_env.save(tmp.name)
        tmp_path = tmp.name

    # Create evaluation vec_env and load training stats
    eval_factory = make_env_factory(graph, config, mode="eval")
    eval_vec_env = make_vec_env(eval_factory, n_envs=1)
    eval_vec_env = VecNormalize.load(tmp_path, eval_vec_env)  # Load from file
    eval_vec_env.training = False
    eval_vec_env.norm_reward = False

    # Verify obs_rms stats match (Dict observation space -> dict of RunningMeanStd)
    train_obs_rms = train_vec_env.obs_rms
    eval_obs_rms = eval_vec_env.obs_rms

    assert set(train_obs_rms.keys()) == set(eval_obs_rms.keys()), "obs_rms keys mismatch"

    for key in train_obs_rms.keys():
        np.testing.assert_allclose(
            train_obs_rms[key].mean,
            eval_obs_rms[key].mean,
            rtol=1e-5,
            err_msg=f"obs_rms[{key}].mean mismatch between train and eval"
        )
        np.testing.assert_allclose(
            train_obs_rms[key].var,
            eval_obs_rms[key].var,
            rtol=1e-5,
            err_msg=f"obs_rms[{key}].var mismatch between train and eval"
        )
        np.testing.assert_allclose(
            train_obs_rms[key].count,
            eval_obs_rms[key].count,
            rtol=1e-5,
            err_msg=f"obs_rms[{key}].count mismatch between train and eval"
        )

    train_vec_env.close()
    eval_vec_env.close()


def test_vecnormalize_ret_rms_sync():
    """Training and evaluation VecNormalize should have identical ret_rms after sync."""
    config = load_config()
    graph = load_graph(GRAPH_PATH)

    train_vec_env = make_normalized_vec_env(graph, config, n_envs=2, mode="train")

    # Simulate training to populate ret_rms
    for _ in range(10):
        train_vec_env.reset()
        train_vec_env.step([0, 0])

    # Save training stats to file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        train_vec_env.save(tmp.name)
        tmp_path = tmp.name

    # Load into eval env
    eval_factory = make_env_factory(graph, config, mode="eval")
    eval_vec_env = make_vec_env(eval_factory, n_envs=1)
    eval_vec_env = VecNormalize.load(tmp_path, eval_vec_env)
    eval_vec_env.training = False
    eval_vec_env.norm_reward = False

    np.testing.assert_allclose(
        train_vec_env.ret_rms.mean,
        eval_vec_env.ret_rms.mean,
        rtol=1e-5,
        err_msg="ret_rms.mean mismatch"
    )
    np.testing.assert_allclose(
        train_vec_env.ret_rms.var,
        eval_vec_env.ret_rms.var,
        rtol=1e-5,
        err_msg="ret_rms.var mismatch"
    )
    np.testing.assert_allclose(
        train_vec_env.ret_rms.count,
        eval_vec_env.ret_rms.count,
        rtol=1e-5,
        err_msg="ret_rms.count mismatch"
    )

    train_vec_env.close()
    eval_vec_env.close()


def test_vecnormalize_clip_reward_applied():
    """VecNormalize should clip rewards during training but not evaluation."""
    config = load_config()
    graph = load_graph(GRAPH_PATH)

    train_vec_env = make_normalized_vec_env(graph, config, n_envs=1, mode="train")
    clip_reward = train_vec_env.clip_reward

    # During training, rewards should be clipped
    train_vec_env.training = True
    train_vec_env.norm_reward = True

    # Save and load for eval
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        train_vec_env.save(tmp.name)
        tmp_path = tmp.name

    eval_factory = make_env_factory(graph, config, mode="eval")
    eval_vec_env = make_vec_env(eval_factory, n_envs=1)
    eval_vec_env = VecNormalize.load(tmp_path, eval_vec_env)
    eval_vec_env.training = False
    eval_vec_env.norm_reward = False

    # During eval, rewards should NOT be clipped (norm_reward=False)
    assert eval_vec_env.norm_reward == False, "Evaluation should have norm_reward=False"
    assert train_vec_env.norm_reward == True, "Training should have norm_reward=True"

    train_vec_env.close()
    eval_vec_env.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
