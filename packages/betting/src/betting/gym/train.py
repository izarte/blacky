import os
import sys

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from betting.gym.environment import BlackjackEnv

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def train_blackjack_agent(
    models_path,
    logs_path,
    tensoboard_logs_path,
    hands_per_check: int = 1,
    num_decks: int = 4,
):
    seed = 42
    # Create environment
    env = BlackjackEnv(hands_per_check=hands_per_check, num_decks=num_decks)
    env = Monitor(env)

    # Create vectorized environment for training
    vec_env = make_vec_env(
        lambda: BlackjackEnv(hands_per_check=hands_per_check, num_decks=num_decks),
        n_envs=8,
        seed=seed,
    )

    # Create evaluation environment
    eval_env = Monitor(
        BlackjackEnv(hands_per_check=hands_per_check, num_decks=num_decks)
    )

    # Define model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=tensoboard_logs_path,
        seed=seed,
        device="cpu",
    )

    # Create callbacks
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        eval_freq=10000,
        best_model_save_path=models_path,
        log_path=logs_path,
        verbose=1,
    )

    # Train the model
    total_timesteps = 10000000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="blackjack_ppo",
        progress_bar=True,
    )

    # Save the final model
    model.save(models_path / "blackjack_ppo_final")

    print("Training completed!")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=500,
        deterministic=True,
    )
    print(f"🏆 Best model mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return model


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)

    # Train the agent
    trained_model = train_blackjack_agent()
