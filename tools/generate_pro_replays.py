#!/usr/bin/env python3
"""Generate pro replays by running fischly's binary CNN in TetrisSim.

Supports two modes:
  1. Binary-only (default): fischly CNN picks placements (good at line clears)
  2. Monoblock-aware (--mono-weight): CNN + monoblock heuristic picks placements
     This teaches the color CNN to cluster same-type pieces for TNT square bonuses.

Output: numpy .npz file with arrays:
  - states: (N, 20, 10) int8 — color board before placement
  - next_states: (N, 20, 10) int8 — color board after placement
  - rewards: (N,) float32 — reward for the placement (includes adjacency bonus)
  - dones: (N,) bool — whether the game ended

Usage:
    .venv/bin/python3 tools/generate_pro_replays.py models/good-cnn-2.pt --games 500
    .venv/bin/python3 tools/generate_pro_replays.py models/good-cnn-2.pt --games 500 --mono-weight 50
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.board_evaluator import BoardEvaluator
from src.ai.monoblock import find_best_placement_combined
from src.game.pieces import PieceType, ROTATION_COUNT, get_cells
from src.game.tetris_sim import TetrisSim, BOARD_COLS, BOARD_ROWS

# Match fischly's reward scale
REWARD_SCALE = {0: 0, 1: 50, 2: 150, 3: 300, 4: 800}
GAME_OVER_REWARD = -5000
MONOSQUARE_SCALED = 2000
MULTISQUARE_SCALED = 1000
ADJACENCY_REWARD_PER_PAIR = 25


def color_adjacency_bonus(board_before, board_after):
    """Count new same-type adjacent pairs created by a placement."""
    def _count_adj(board):
        occupied = board >= 0
        h = (board[:, :-1] == board[:, 1:]) & occupied[:, :-1] & occupied[:, 1:]
        v = (board[:-1, :] == board[1:, :]) & occupied[:-1, :] & occupied[1:, :]
        return int(h.sum()) + int(v.sum())
    return _count_adj(board_after) - _count_adj(board_before)


def generate_replays(evaluator: BoardEvaluator, num_games: int, device: str,
                     mono_weight: float = 0.0):
    states = []
    next_states = []
    rewards = []
    dones = []

    total_lines = 0
    total_pieces = 0
    total_mono = 0
    total_multi = 0

    use_mono = mono_weight > 0
    if use_mono:
        print(f"Using monoblock-aware placement (weight={mono_weight})")
    else:
        print("Using binary-only placement")

    start = time.time()

    for game_idx in range(num_games):
        sim = TetrisSim()
        sim.reset()

        while not sim.game_over:
            # Get color board before placement
            color_before = sim.get_color_board()
            binary_board = sim.board.copy()

            # Pick placement
            if use_mono:
                rot, col, _, _, _ = find_best_placement_combined(
                    evaluator, binary_board, color_before,
                    sim.current_piece, mono_weight,
                )
            else:
                rot, col, score = evaluator.find_best_placement(
                    binary_board, sim.current_piece,
                )

            # Step the sim
            _, sim_reward, done, info = sim.step(rot, col)

            # Get color board after placement
            color_after = sim.get_color_board()

            # Compute reward with adjacency bonus
            if done:
                reward = float(GAME_OVER_REWARD)
            else:
                lines = info.get("lines", 0)
                nm = info.get("new_mono_squares", 0)
                nmulti = info.get("new_multi_squares", 0)
                reward = float(REWARD_SCALE.get(lines, lines * 100))
                reward += nm * MONOSQUARE_SCALED + nmulti * MULTISQUARE_SCALED
                reward += color_adjacency_bonus(color_before, color_after) * ADJACENCY_REWARD_PER_PAIR

            states.append(color_before)
            next_states.append(color_after)
            rewards.append(reward)
            dones.append(done)

        total_lines += sim.lines_cleared
        total_pieces += sim.pieces_placed
        total_mono += sim.mono_squares
        total_multi += sim.multi_squares

        if (game_idx + 1) % 50 == 0:
            elapsed = time.time() - start
            avg_lines = total_lines / (game_idx + 1)
            avg_pieces = total_pieces / (game_idx + 1)
            print(f"Game {game_idx + 1}/{num_games} | "
                  f"avg lines={avg_lines:.1f} pieces={avg_pieces:.0f} | "
                  f"mono={total_mono} multi={total_multi} | "
                  f"{len(states)} transitions | {elapsed:.0f}s")

    elapsed = time.time() - start
    avg_l = total_lines / num_games
    avg_mono = total_mono / num_games
    avg_multi = total_multi / num_games
    tnt = avg_l + avg_mono * 30 + avg_multi * 15
    print(f"\nDone: {num_games} games, {len(states)} transitions in {elapsed:.0f}s")
    print(f"Avg: {avg_l:.1f} lines, {total_pieces/num_games:.0f} pieces/game")
    print(f"Squares: {total_mono} mono ({avg_mono:.3f}/game), {total_multi} multi ({avg_multi:.3f}/game)")
    print(f"Estimated TNT score: {tnt:.1f}/game")

    return (
        np.array(states, dtype=np.int8),
        np.array(next_states, dtype=np.int8),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=bool),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate pro replays from fischly CNN")
    parser.add_argument("weights", help="Path to fischly CNN weights (e.g. models/good-cnn-2.pt)")
    parser.add_argument("--games", type=int, default=500, help="Number of games to play")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("-o", "--output", default="pro-replays/expert.npz",
                        help="Output .npz file path")
    parser.add_argument("--mono-weight", type=float, default=0.0,
                        help="Monoblock heuristic weight (0=binary-only, 50=moderate, 200=aggressive)")
    args = parser.parse_args()

    print(f"Loading fischly CNN from {args.weights}")
    evaluator = BoardEvaluator(args.weights, device=args.device)

    # Quick sanity check
    empty_score = evaluator.evaluate(np.zeros((20, 10), dtype=bool))
    print(f"Empty board score: {empty_score:.2f}")

    states, next_states, rewards, dones = generate_replays(
        evaluator, args.games, args.device, mono_weight=args.mono_weight
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=states,
        next_states=next_states,
        rewards=rewards,
        dones=dones,
    )
    print(f"Saved {len(states)} transitions to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
