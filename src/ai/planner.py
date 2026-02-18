"""Move planner: searches all possible piece placements and selects the best.

Generates every valid (rotation, column) combination, simulates the drop,
evaluates the resulting board with the heuristic, and returns the best
placement. Supports optional 2-ply lookahead using the next piece.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..game.board import Board, Cell
from ..game.pieces import PieceType, ROTATION_COUNT, get_cells
from .heuristic import HeuristicEvaluator


@dataclass
class Placement:
    """A specific placement of a piece on the board."""

    piece: PieceType
    rotation: int     # 0-3
    column: int       # leftmost column where piece is placed
    row: int          # row where piece lands
    score: float      # heuristic evaluation score


class MovePlanner:
    """Searches all possible placements and selects the best one."""

    def __init__(self, evaluator: HeuristicEvaluator):
        self.evaluator = evaluator

    def find_best_placement(
        self,
        board: Board,
        piece: PieceType,
        next_piece: PieceType | None = None,
    ) -> Placement | None:
        """Find the highest-scoring placement for the current piece.

        If next_piece is provided, does 2-ply lookahead: for each placement
        of the current piece, finds the best placement of the next piece and
        uses the combined score.
        """
        candidates = self._generate_all_placements(board, piece)
        if not candidates:
            return None

        if next_piece is None:
            return max(candidates, key=lambda p: p.score)

        # 2-ply lookahead
        best: Placement | None = None
        best_score = float("-inf")

        for placement in candidates:
            # Simulate placing the current piece
            simulated, _ = self._simulate_placement(board, placement)

            # Find best placement for next piece on the simulated board
            next_candidates = self._generate_all_placements(simulated, next_piece)
            if next_candidates:
                best_next = max(next_candidates, key=lambda p: p.score)
                combined = placement.score * 0.6 + best_next.score * 0.4
            else:
                # Next piece has no valid placements -> game over
                combined = placement.score - 1000

            if combined > best_score:
                best_score = combined
                best = Placement(
                    piece=placement.piece,
                    rotation=placement.rotation,
                    column=placement.column,
                    row=placement.row,
                    score=combined,
                )

        return best

    def _generate_all_placements(
        self, board: Board, piece: PieceType
    ) -> list[Placement]:
        """Generate every valid placement and evaluate each."""
        placements = []
        num_rotations = ROTATION_COUNT[piece]

        for rot in range(num_rotations):
            cells = get_cells(piece, rot)
            min_col = min(c for _, c in cells)
            max_col = max(c for _, c in cells)

            # Try every column where the piece fits
            for col in range(-min_col, board.WIDTH - max_col):
                row = self._find_landing_row(board, cells, col)
                if row is None:
                    continue  # Piece doesn't fit (above the board)

                # Simulate the drop and evaluate
                simulated, lines_cleared = self._simulate_drop(
                    board, cells, col, row
                )
                score = self.evaluator.evaluate(simulated)
                # Bonus for clearing lines (already captured in evaluator,
                # but give extra weight to immediate clears)
                score += lines_cleared * 0.5

                placements.append(Placement(piece, rot, col, row, score))

        return placements

    @staticmethod
    def _find_landing_row(
        board: Board, cells: list[tuple[int, int]], col: int
    ) -> int | None:
        """Find the row where a piece lands when dropped at the given column.

        Returns the top-left row offset where the piece rests, or None if
        the piece can't be placed (collision at row 0).
        """
        # Start from the top and move down until collision
        for start_row in range(board.HEIGHT + 1):
            for dr, dc in cells:
                r = start_row + dr
                c = col + dc
                if r >= board.HEIGHT:
                    return start_row - 1 if start_row > 0 else None
                if board.cells[r][c].occupied:
                    return start_row - 1 if start_row > 0 else None
        # Piece fell all the way (shouldn't happen with a proper board)
        max_dr = max(dr for dr, _ in cells)
        return board.HEIGHT - 1 - max_dr

    @staticmethod
    def _simulate_drop(
        board: Board, cells: list[tuple[int, int]], col: int, row: int
    ) -> tuple[Board, int]:
        """Place a piece on the board and clear any complete lines.

        Returns (new_board, lines_cleared).
        """
        # Deep copy the board cells
        new_cells = []
        for r in range(board.HEIGHT):
            new_row = []
            for c in range(board.WIDTH):
                old = board.cells[r][c]
                new_cell = Cell(old.raw)
                new_row.append(new_cell)
            new_cells.append(new_row)

        # Place the piece
        for dr, dc in cells:
            r = row + dr
            c = col + dc
            if 0 <= r < board.HEIGHT and 0 <= c < board.WIDTH:
                new_cells[r][c].color = 1  # Generic color
                new_cells[r][c].occupied = True

        new_board = Board(new_cells)
        cleared_board, lines = new_board.clear_lines()
        return cleared_board, lines

    def _simulate_placement(
        self, board: Board, placement: Placement
    ) -> tuple[Board, int]:
        """Simulate a Placement on the board."""
        cells = get_cells(placement.piece, placement.rotation)
        return self._simulate_drop(board, cells, placement.column, placement.row)
