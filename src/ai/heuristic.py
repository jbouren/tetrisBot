"""Heuristic evaluation function for Tetris board positions.

Scores a board position using a weighted sum of features.
Higher scores are better. Initial weights are based on the
Yiyuan Lee genetic algorithm results for standard Tetris,
extended with The New Tetris-specific square mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..game.squares import SquareType, count_square_potential, detect_squares

if TYPE_CHECKING:
    from ..game.board import Board


@dataclass
class Weights:
    """Tunable weights for the heuristic evaluation."""

    # Standard Tetris features
    aggregate_height: float = -0.510066
    complete_lines: float = 0.760666
    holes: float = -0.35663
    bumpiness: float = -0.184483
    max_height: float = -0.3

    # The New Tetris-specific features
    gold_squares: float = 5.0
    silver_squares: float = 3.0
    square_potential: float = 1.0

    # Well depth (positive = good for I-piece Tetrises)
    well_depth: float = 0.2

    # Height danger zone penalty (triggers when max height > 15)
    danger_zone_penalty: float = -2.0


DEFAULT_WEIGHTS = Weights()


class HeuristicEvaluator:
    """Scores board positions using weighted features."""

    def __init__(self, weights: Weights | None = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def evaluate(self, board: Board) -> float:
        """Score a board position. Higher is better."""
        w = self.weights
        score = 0.0

        # Standard features
        score += w.aggregate_height * board.aggregate_height()
        score += w.complete_lines * board.count_complete_lines()
        score += w.holes * board.count_holes()
        score += w.bumpiness * board.bumpiness()
        score += w.max_height * board.max_height()

        # Danger zone: extra penalty when stack is very tall
        if board.max_height() > 15:
            score += w.danger_zone_penalty * (board.max_height() - 15)

        # The New Tetris: square detection
        squares = detect_squares(board)
        gold_count = sum(1 for _, _, t in squares if t == SquareType.GOLD)
        silver_count = sum(1 for _, _, t in squares if t == SquareType.SILVER)
        score += w.gold_squares * gold_count
        score += w.silver_squares * silver_count
        score += w.square_potential * count_square_potential(board)

        # Well detection: reward a single deep well for I-piece Tetrises
        score += w.well_depth * self._well_score(board)

        return score

    @staticmethod
    def _well_score(board: Board) -> float:
        """Score well formations on the board.

        A "well" is a column that is lower than both neighbors.
        A single deep well is good (for I-piece Tetrises).
        Multiple wells are bad.
        """
        heights = board.column_heights()
        wells = []

        for c in range(board.WIDTH):
            left_h = heights[c - 1] if c > 0 else 20
            right_h = heights[c + 1] if c < board.WIDTH - 1 else 20
            col_h = heights[c]
            depth = min(left_h, right_h) - col_h
            if depth > 0:
                wells.append(depth)

        if len(wells) == 0:
            return 0.0
        if len(wells) == 1:
            # Single well: reward depth (good for Tetrises)
            return wells[0] * 1.0
        # Multiple wells: penalize (dispersed gaps are bad)
        return -sum(wells) * 0.5
