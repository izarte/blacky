from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from betting.gym.environment import BlackjackEnv


def evaluate(
    model_path: str,
    episodes: int = 500,
    hands_per_check: int = 100,
    num_decks: int = 4,
):
    eval_env = Monitor(
        BlackjackEnv(hands_per_check=hands_per_check, num_decks=num_decks)
    )

    model = PPO.load(model_path, device="cpu")

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
    )
    return mean_reward, std_reward
