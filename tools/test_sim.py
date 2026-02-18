#!/usr/bin/env python3
"""Test and benchmark the TetrisSim pure Python simulator.

Runs random games to verify correctness and measure performance.

Usage:
    .venv/bin/python3 tools/test_sim.py
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.tetris_sim import TetrisSim
from src.game.pieces import ROTATION_COUNT, get_width


def play_random_game(sim: TetrisSim, seed: int | None = None) -> dict:
    """Play one game with random valid actions. Returns stats."""
    rng = random.Random(seed)
    sim.reset()

    while not sim.game_over:
        actions = sim.get_valid_actions()
        if not actions:
            break
        rotation, column = rng.choice(actions)
        sim.step(rotation, column)

    return {
        "pieces": sim.pieces_placed,
        "lines": sim.lines_cleared,
        "score": sim.score,
        "mono_squares": sim.mono_squares,
        "multi_squares": sim.multi_squares,
    }


def main():
    n_games = 100
    print(f"=== TetrisSim Benchmark: {n_games} random games ===\n")

    sim = TetrisSim()
    results = []

    t0 = time.perf_counter()
    for i in range(n_games):
        stats = play_random_game(sim, seed=i)
        results.append(stats)
    elapsed = time.perf_counter() - t0

    pieces = [r["pieces"] for r in results]
    lines = [r["lines"] for r in results]
    scores = [r["score"] for r in results]
    monos = [r["mono_squares"] for r in results]
    multis = [r["multi_squares"] for r in results]

    print(f"Games played:    {n_games}")
    print(f"Time elapsed:    {elapsed:.3f}s")
    print(f"Games/second:    {n_games / elapsed:.1f}")
    print()
    print(f"Pieces per game:  min={min(pieces)} avg={sum(pieces)/len(pieces):.1f} max={max(pieces)}")
    print(f"Lines per game:   min={min(lines)} avg={sum(lines)/len(lines):.1f} max={max(lines)}")
    print(f"Score per game:   min={min(scores)} avg={sum(scores)/len(scores):.1f} max={max(scores)}")
    print(f"Mono squares:     total={sum(monos)} across {n_games} games ({sum(monos)/n_games:.2f}/game)")
    print(f"Multi squares:    total={sum(multis)} across {n_games} games ({sum(multis)/n_games:.2f}/game)")

    # Show a few sample final boards
    print("\n=== Sample Games ===\n")
    for game_id in [0, 1, 2]:
        sim_sample = TetrisSim(seed=game_id)
        play_random_game(sim_sample, seed=game_id + 1000)
        print(f"--- Game {game_id} ({sim_sample.pieces_placed} pieces, "
              f"{sim_sample.lines_cleared} lines, "
              f"{sim_sample.mono_squares}M+{sim_sample.multi_squares}X squares) ---")
        print(sim_sample.render())
        print()

    # Stress test: 10k games for throughput
    print("=== Stress test: 10,000 games ===")
    sim2 = TetrisSim()
    t1 = time.perf_counter()
    for i in range(10_000):
        play_random_game(sim2, seed=i)
    t2 = time.perf_counter()
    print(f"10,000 games in {t2 - t1:.2f}s ({10_000 / (t2 - t1):.0f} games/sec)")


if __name__ == "__main__":
    main()
