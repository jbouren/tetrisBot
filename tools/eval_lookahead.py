#!/usr/bin/env python3
"""Evaluate CNN evaluator with lookahead vs baseline in simulation.

Runs N games in the simulator (no emulator needed) comparing:
  - Baseline: depth=0 (CNN only, same as find_best_placement)
  - Lookahead depth 1, 2, 3

Usage:
  .venv/bin/python3 tools/eval_lookahead.py models/good-cnn-2.pt
  .venv/bin/python3 tools/eval_lookahead.py models/good-cnn-2.pt --games 50 --depth 2
  .venv/bin/python3 tools/eval_lookahead.py models/good-cnn-2.pt --depth 1 2 3  # compare all
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai.board_evaluator import BoardEvaluator
from src.ai.lookahead import find_best_placement_lookahead
from src.game.tetris_sim import TetrisSim


def run_games(evaluator, depth: int, beam_width: int, n_games: int, seed_offset: int = 0
              ) -> list[dict]:
    """Run n_games in simulation with the given lookahead depth."""
    results = []
    for game_idx in range(n_games):
        t_game = time.time()
        sim = TetrisSim(seed=seed_offset + game_idx)
        sim.reset()

        while not sim.game_over:
            board = sim.board
            current_piece = sim.current_piece
            # preview: next_pieces gives the upcoming pieces after current
            preview = list(sim.next_pieces)

            if depth == 0:
                rot, col, score = evaluator.find_best_placement(board, current_piece)
            else:
                rot, col, score = find_best_placement_lookahead(
                    evaluator, board, current_piece,
                    preview=preview[:depth],
                    depth=depth,
                    beam_width=beam_width,
                )

            _, _, done, _ = sim.step(rot, col)
            if done:
                break

        elapsed_game = time.time() - t_game
        results.append({
            "lines": sim.lines_cleared,
            "pieces": sim.pieces_placed,
        })
        print(f"  game {game_idx+1:>2}/{n_games}  lines={sim.lines_cleared:>4}  "
              f"pieces={sim.pieces_placed:>4}  {elapsed_game:.1f}s", flush=True)

    return results


def print_stats(label: str, results: list[dict], elapsed: float):
    lines = [r["lines"] for r in results]
    pieces = [r["pieces"] for r in results]
    n = len(results)
    print(f"\n{'='*55}", flush=True)
    print(f"  {label}  ({n} games, {elapsed:.1f}s total, {elapsed/n:.2f}s/game)")
    print(f"{'='*55}")
    print(f"  Lines:  avg={np.mean(lines):.1f}  median={np.median(lines):.0f}"
          f"  best={max(lines)}  worst={min(lines)}")
    print(f"  Pieces: avg={np.mean(pieces):.1f}  best={max(pieces)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN lookahead vs baseline in sim")
    parser.add_argument("weights", help="Path to CNN weights (.pt)")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per configuration (default 20)")
    parser.add_argument("--depth", type=int, nargs="+", default=[0, 1, 2],
                        help="Lookahead depths to evaluate (default: 0 1 2)")
    parser.add_argument("--beam-width", type=int, default=10,
                        help="Beam width for pruning (default 10)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline (depth=0)")
    args = parser.parse_args()

    print(f"Loading evaluator from {args.weights} (device={args.device})...", flush=True)
    evaluator = BoardEvaluator(args.weights, device=args.device)

    depths = [0] if args.baseline_only else args.depth

    for depth in depths:
        label = f"Depth {depth} (baseline)" if depth == 0 else f"Depth {depth} lookahead"
        print(f"\nRunning {args.games} games: {label}...", flush=True)
        t0 = time.time()
        results = run_games(evaluator, depth, args.beam_width, args.games, seed_offset=42)
        elapsed = time.time() - t0
        print_stats(label, results, elapsed)


if __name__ == "__main__":
    main()
