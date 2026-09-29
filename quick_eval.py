from src.evaluation.run_scenarios import generate_scenarios, evaluate_scenarios, _make_env
from src.agents.training_utils import load_graph, load_config, GRAPH_PATH, TRAFFIC_PROFILE_PATH, CLIMATOLOGY_PATH
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import networkx as nx

config = load_config()
graph = load_graph(GRAPH_PATH)
env_ctor_args = (str(TRAFFIC_PROFILE_PATH), str(CLIMATOLOGY_PATH))

# Generate just a few test scenarios
scenarios = generate_scenarios(graph, env_ctor_args, config, num_scenarios=10, num_flood_scenarios=5, seed=999)
print(f'Natural: {len(scenarios["natural"])}, Flood: {len(scenarios["flood_stratified"])}')

# Load model
model = MaskablePPO.load('models/ppo_full/model.zip')
rl_vec_env = DummyVecEnv([lambda: _make_env(graph, env_ctor_args, config)])
rl_vec_env = VecNormalize.load('models/ppo_full/vecnormalize.pkl', rl_vec_env)
rl_vec_env.training = False
rl_vec_env.norm_reward = False
rl_base_env = rl_vec_env.venv.envs[0]

# Test on first flood scenario
scenario = scenarios['flood_stratified'][0]
print(f'Testing flood scenario: {scenario["seed"]}, flood_day={scenario["is_flood_day"]}')

# Run RL
from src.evaluation.run_scenarios import _evaluate_rl_scenario
result = _evaluate_rl_scenario(model, rl_vec_env, rl_base_env, scenario)
print(f'RL result: success={result["success"]}, reward={result["reward"]}, time={result["realized_travel_time"]}')

# Run masked random
from src.evaluation.run_scenarios import _run_env_policy
env = _make_env(graph, env_ctor_args, config)
rng = np.random.default_rng(0)
mr_result = _run_env_policy(env, scenario, lambda e, o: int(rng.choice(np.flatnonzero(e.action_masks()))))
print(f'Masked random: success={mr_result["success"]}, reward={mr_result["reward"]}')
env.close()

# Run static
env = _make_env(graph, env_ctor_args, config)
from src.evaluation.run_scenarios import _static_action_selector
static_result = _run_env_policy(env, scenario, _static_action_selector)
print(f'Static: success={static_result["success"]}, reward={static_result["reward"]}')
env.close()

rl_vec_env.close()
