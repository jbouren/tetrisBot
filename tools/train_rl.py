#!/usr/bin/env python3
"""Train a PPO agent to play Tetris using TetrisSim."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.game.tetris_env import TetrisEnv


def make_env(seed=None):
    def _init():
        return TetrisEnv(seed=seed)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on TetrisSim")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--output", default="models/ppo_tetris", help="Model save path (without extension)")
    parser.add_argument("--resume", default=None, help="Path to model to resume training from")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    print(f"Creating {args.envs} parallel environments...")
    env = SubprocVecEnv([make_env(seed=i) for i in range(args.envs)])

    # Eval env for periodic evaluation
    eval_env = TetrisEnv(seed=42)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log="runs/", device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log="runs/",
            device="cpu",
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path="runs/eval/",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"Training for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    save_path = f"{args.output}_{args.timesteps // 1000}k"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
