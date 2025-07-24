from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from betting.gym.environment import BlackjackEnv


def evaluate(model_path: str, episodes: int = 500):
    eval_env = Monitor(BlackjackEnv())

    model = PPO.load(model_path)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
    )
    return mean_reward, std_reward
