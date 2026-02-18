#!/usr/bin/env python3
"""Evaluate a trained RL model against a random baseline."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO

from src.game.tetris_env import TetrisEnv


def run_games(env, model, n_games, deterministic=True):
    """Run n_games and collect stats."""
    results = []
    for i in range(n_games):
        obs, _ = env.reset(seed=1000 + i)
        total_reward = 0
        while True:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        results.append({
            "pieces": env.sim.pieces_placed,
            "lines": env.sim.lines_cleared,
            "score": env.sim.score,
            "reward": total_reward,
        })
    return results


def print_stats(label, results):
    pieces = [r["pieces"] for r in results]
    lines = [r["lines"] for r in results]
    scores = [r["score"] for r in results]
    rewards = [r["reward"] for r in results]

    print(f"\n{'='*50}")
    print(f"  {label} ({len(results)} games)")
    print(f"{'='*50}")
    print(f"  Pieces:  avg={np.mean(pieces):.1f}  best={max(pieces)}  worst={min(pieces)}")
    print(f"  Lines:   avg={np.mean(lines):.1f}  best={max(lines)}  worst={min(lines)}")
    print(f"  Score:   avg={np.mean(scores):.1f}  best={max(scores)}  worst={min(scores)}")
    print(f"  Reward:  avg={np.mean(rewards):.1f}  best={max(rewards):.1f}  worst={min(rewards):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Tetris agent")
    parser.add_argument("model", help="Path to trained model (.zip)")
    parser.add_argument("--games", type=int, default=100, help="Number of evaluation games")
    parser.add_argument("--show", action="store_true", help="Print board from best game")
    args = parser.parse_args()

    env = TetrisEnv()

    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model, device="cpu")

    print(f"Running {args.games} games with trained agent...")
    trained_results = run_games(env, model, args.games)
    print_stats("Trained Agent", trained_results)

    print(f"\nRunning {args.games} games with random agent...")
    random_results = run_games(env, None, args.games)
    print_stats("Random Baseline", random_results)

    trained_avg = np.mean([r["lines"] for r in trained_results])
    random_avg = np.mean([r["lines"] for r in random_results])
    if random_avg > 0:
        print(f"\nTrained agent clears {trained_avg/random_avg:.1f}x more lines than random")
    else:
        print(f"\nTrained agent avg lines: {trained_avg:.1f} vs random: {random_avg:.1f}")

    if args.show:
        # Replay the best game
        best_idx = max(range(len(trained_results)), key=lambda i: trained_results[i]["lines"])
        print(f"\n--- Best game (game #{best_idx}, {trained_results[best_idx]['lines']} lines) ---")
        obs, _ = env.reset(seed=1000 + best_idx)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        print(env.sim.render())

    env.close()


if __name__ == "__main__":
    main()
