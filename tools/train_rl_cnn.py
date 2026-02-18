#!/usr/bin/env python3
"""Train a PPO agent with CNN policy to play Tetris using TetrisSim."""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.game.tetris_env_cnn import TetrisEnvCNN


class TetrisCNN(BaseFeaturesExtractor):
    """Custom CNN for the 20x10 Tetris board with piece channels.

    Uses small 3x3 kernels suitable for our compact board.
    Architecture:
        Conv2d(29, 64, 3, padding=1) -> ReLU
        Conv2d(64, 128, 3, padding=1) -> ReLU
        MaxPool2d(2)  -> 10x5
        Conv2d(128, 128, 3, padding=1) -> ReLU
        Conv2d(128, 64, 3, padding=1) -> ReLU
        Flatten -> 3200
        Linear(3200, features_dim) -> ReLU
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[2]  # (H, W, C) -> C=29

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20x10 -> 10x5
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, 20, 10)
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # SB3 passes (batch, H, W, C), need (batch, C, H, W) for PyTorch
        x = observations.permute(0, 3, 1, 2).float()
        return self.linear(self.cnn(x))


def make_env(seed=None):
    def _init():
        return TetrisEnvCNN(seed=seed)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with CNN on TetrisSim")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--output", default="models/ppo_tetris_cnn",
                        help="Model save path (without extension)")
    parser.add_argument("--resume", default=None,
                        help="Path to model to resume training from")
    parser.add_argument("--features-dim", type=int, default=256,
                        help="Feature extractor output dimension")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size for training updates")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per env per rollout")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of epochs per training update")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    print(f"Creating {args.envs} parallel environments...")
    env = SubprocVecEnv([make_env(seed=i) for i in range(args.envs)])

    eval_env = TetrisEnvCNN(seed=42)

    policy_kwargs = dict(
        features_extractor_class=TetrisCNN,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log="runs/",
                         device=args.device)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            verbose=1,
            tensorboard_log="runs/",
            device=args.device,
            policy_kwargs=policy_kwargs,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_cnn/",
        log_path="runs/eval_cnn/",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Policy parameters: {total_params:,} total, {trainable:,} trainable")
    print(f"Training for {args.timesteps:,} timesteps on {args.device}...")

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    save_path = f"{args.output}_{args.timesteps // 1000}k"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
