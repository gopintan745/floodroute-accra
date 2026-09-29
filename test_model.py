from src.agents.training_utils import load_graph, load_config, GRAPH_PATH, TRAFFIC_PROFILE_PATH, CLIMATOLOGY_PATH
from src.env.accra_routing_env import AccraRoutingEnv
from sb3_contrib import MaskablePPO
import numpy as np

config = load_config()
graph = load_graph(GRAPH_PATH)
env = AccraRoutingEnv(graph, str(TRAFFIC_PROFILE_PATH), str(CLIMATOLOGY_PATH), config, mode='eval')
model = MaskablePPO.load('models/ppo_full/model.zip')

# Quick test: reset and get one action
obs, info = env.reset(seed=42)
print(f'Reset: origin={info["origin"]}, dest={info["destination"]}, flood_day={info["is_flood_day"]}')
print(f'Obs keys: {obs.keys()}')
print(f'Action masks: {env.action_masks()}')

action, _ = model.predict(obs, deterministic=True, action_masks=np.asarray(env.action_masks()))
print(f'Predicted action: {action}')
