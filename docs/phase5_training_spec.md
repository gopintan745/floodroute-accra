# Phase 5 Spec: Agent Training (`train_smallgraph_sanity.py` + `train_ppo.py`)

Handoff spec for training. Read alongside `docs/phase4_hazard_simulator_spec.md`
and `docs/phase4_env_spec.md` — this phase trains against the environment
and masking scheme those specs define.

## Housekeeping: rename `train_dqn.py`

Rename `src/agents/train_dqn.py` -> `src/agents/train_smallgraph_sanity.py`.
Since the action-space decision settled on `MaskablePPO` everywhere (DQN
has no clean masking support for the variable-legal-actions problem), a
file named `train_dqn.py` containing `MaskablePPO` code would be actively
misleading to read later. The *purpose* of this script is unchanged from
the original roadmap — a fast, cheap sanity check before committing to a
full training run — only the algorithm name in the filename was stale.

## Decision 1: small-subgraph extraction method

Use `networkx.ego_graph(full_graph, center_node, radius=N)` to carve out
a real corner of the actual study-area graph, rather than a synthetic toy
graph. `radius` in hops, not meters — grow outward until under a node
budget (aim for well under 50 nodes, per the original roadmap).

```python
center_node = config.training.sanity_check.center_node  # e.g. Kaneshie market node
subgraph_radius = config.training.sanity_check.subgraph_radius
subgraph = nx.ego_graph(full_graph, center_node, radius=subgraph_radius)
```

Picking `center_node`: a natural default is the Kaneshie market node
itself, or the midpoint between a representative origin/destination pair.
Deterministic and reproducible — same subgraph every run given the same
config, which matters for comparing sanity-check runs across code changes.

## Decision 2: reward scaling via `VecNormalize`

Rewards are raw seconds — single-edge costs land around -100 to -600,
full episodes can accumulate several thousand in negative reward.
Unscaled, this is a common source of unstable PPO training (large-magnitude
rewards make the value function's job harder, can cause early gradient
blowups). No new dependency needed — `stable_baselines3.common.vec_env.VecNormalize`
wraps the vectorized env with running-statistics normalization.

```python
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env(lambda: AccraRoutingEnv(...), n_envs=n_envs)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
```

Treat this as a default included from the start in BOTH training scripts,
not something to bolt on later after debugging an unstable run — the
instability it causes can be hard to distinguish from a genuine reward-design
bug otherwise (worth remembering given how many bugs in this project so far
have been "looks like it's working, but the numbers are subtly wrong").

**Important**: `VecNormalize`'s running statistics must be saved alongside
the model checkpoint (`vec_env.save(path)`) and reloaded before evaluation
— a model evaluated through a differently-normalized env will produce
meaningless results. Flag this explicitly in `evaluation/run_scenarios.py`
when that's built (Phase 6).

## `src/agents/train_smallgraph_sanity.py`

```python
def main():
    config = load_config(...)
    full_graph = load_graph(...)  # road_graph_full.graphml

    subgraph = nx.ego_graph(
        full_graph,
        config.training.sanity_check.center_node,
        radius=config.training.sanity_check.subgraph_radius,
    )

    def make_env():
        return AccraRoutingEnv(subgraph, traffic_profile, climatology, config, mode="train")

    vec_env = make_vec_env(make_env, n_envs=4)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = MaskablePPO("MultiInputPolicy", vec_env, verbose=1,
                         tensorboard_log=RESULTS_LOGS_DIR)
    model.learn(total_timesteps=config.training.sanity_check.total_timesteps)

    model.save(MODELS_DIR / "sanity_smallgraph" / "model")
    vec_env.save(MODELS_DIR / "sanity_smallgraph" / "vecnormalize.pkl")

    # --- Sanity checks: print/log these, don't just eyeball the training curve ---
    # 1. Mean reward over N eval episodes beats a masked-random policy
    #    (random.choice among currently-valid actions via action_masks())
    # 2. Mean reward beats static_shortest_path evaluated on the same
    #    subgraph's scenarios (same origin/destination pairs, same hazard
    #    realizations via a fixed-seed HazardSimulator(mode="eval"))
    # 3. Training reward curve (from TensorBoard logs) trends upward and
    #    stabilizes — flat or diverging means stop and debug the env/reward
    #    before touching the full graph
```

## `src/agents/train_ppo.py`

Same structure, full study-area graph, more environments in parallel:

```python
vec_env = make_vec_env(make_env, n_envs=config.training.ppo.n_envs)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
model = MaskablePPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=RESULTS_LOGS_DIR)
model.learn(total_timesteps=config.training.ppo.total_timesteps)
model.save(MODELS_DIR / "ppo_full" / "model")
vec_env.save(MODELS_DIR / "ppo_full" / "vecnormalize.pkl")
```

**Do not run this until `train_smallgraph_sanity.py`'s three checks all
pass.** A reward-design or masking bug is far cheaper to catch on a 50-node
subgraph in minutes than on the full graph after hours of training.

## Config additions (`config.example.yaml`)

```yaml
training:
  sanity_check:
    total_timesteps: 50000
    subgraph_radius: 3        # hops from center_node for ego_graph extraction
    center_node: null         # null = fall back to Kaneshie market node at runtime; set explicitly to override
  ppo:
    n_envs: 4                 # already present from original scaffold — confirm it's still set
```

(`training.ppo.total_timesteps` already exists from the original scaffold
— no change needed there.)

## Testing / sanity checklist

- [ ] **Masked-random beats nothing, sanity-checked agent beats masked-random**:
      confirm the comparison actually runs and the ordering holds
      (random < trained). If the trained agent doesn't clear this bar,
      the problem is almost certainly upstream in the env/reward, not the
      training run itself — don't scale up to the full graph to try to
      fix it with more compute.
- [ ] **Sanity-checked agent vs. `static_shortest_path` on the same subgraph**:
      this is the first real signal (even if only on a small graph) that
      the whole project's premise — adaptive routing beats naive shortest-path
      under uncertainty — might hold. A loss here isn't fatal at this stage,
      but should prompt a closer look before committing to full training.
- [ ] **`VecNormalize` statistics saved and reloadable**: save, reload in a
      fresh process, confirm normalization stats match — this is the kind
      of thing that fails silently (env runs fine, just produces subtly
      wrong evaluation numbers) if skipped.
- [ ] **Training curve inspection**: reward trends upward and plateaus
      within the sanity-check budget; entropy doesn't collapse to near-zero
      too early (a sign the policy stopped exploring prematurely) or stay
      high throughout (a sign it never committed to anything learned).
- [ ] **Full `train_ppo.py` run is not started until all of the above pass.**
