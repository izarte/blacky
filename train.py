# train_ppo_aec.py
import os

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

from environment import BlackjackEnv  # ← your original AEC env

# ── 1.  Register the AEC env directly ────────────────────────────────────
NUM_PLAYERS = 5
ENV_NAME = "blackjack_aec_v0"


def env_creator(_=None):
    return BlackjackEnv(num_players=NUM_PLAYERS)


register_env(ENV_NAME, lambda cfg: PettingZooEnv(env_creator()))

# Grab spaces once so we can declare the policies
test_env = PettingZooEnv(env_creator())
OBS_SPACE = test_env.observation_space
ACT_SPACE = test_env.action_space

# ── 2.  One shared PPO policy for every seat (change if you want) ───────
POLICIES = {"shared": (None, OBS_SPACE, ACT_SPACE, {})}


def policy_map(agent_id, *args, **kwargs):
    return "shared"


# ── 3.  PPO configuration (no aec_to_parallel, no supersuit) ────────────
config = (
    PPOConfig()
    .environment(env=ENV_NAME)
    .env_runners(num_env_runners=2, rollout_fragment_length=200)
    .training(
        train_batch_size=4000,
        minibatch_size=256,
        lr=5e-4,
        num_epochs=10,
        model={"fcnet_hiddens": [256, 256]},
    )
    .multi_agent(
        policies=POLICIES,
        policy_mapping_fn=policy_map,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_duration=5,
        evaluation_config={"explore": False},
    )
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .debugging(log_level="DEBUG")
    .framework(framework="torch")
    .api_stack(
        enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    )
)


# ── 4.  Launch training ─────────────────────────────────────────────────
if __name__ == "__main__":
    ray.init()
    tune.run(
        "PPO",
        name="Blackjack_PPO_AEC",
        stop={"timesteps_total": 2_000_000},
        checkpoint_freq=10,
        config=config.to_dict(),
        verbose=2,
    )
