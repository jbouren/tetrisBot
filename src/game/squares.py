"""4x4 square detection for The New Tetris.

The New Tetris awards bonus points for forming 4x4 squares:
  - Gold square: 4 identical tetrominoes forming a 4x4 block
  - Silver square: 4 different tetrominoes forming a 4x4 block

The game scans the playfield top-to-bottom, left-to-right:
  Pass 1: detect monosquares (gold) and mark matched cells
  Pass 2: detect multisquares (silver) among remaining cells

A valid 4x4 square requires:
  - All 16 cells occupied
  - No cell has the "broken" flag set (broken by a previous line clear)
  - Connection bits form valid intact tetrominoes within the square
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .board import Board


class SquareType(Enum):
    NONE = 0
    SILVER = 1  # 4 different tetrominoes
    GOLD = 2    # 4 identical tetrominoes


def detect_4x4_regions(board: Board) -> list[tuple[int, int, bool]]:
    """Find all 4x4 regions where all 16 cells are occupied and unbroken.

    Returns list of (top_row, left_col, is_single_color).
    """
    results = []
    for r in range(board.HEIGHT - 3):
        for c in range(board.WIDTH - 3):
            all_occupied = True
            all_unbroken = True
            colors = set()

            for dr in range(4):
                for dc in range(4):
                    cell = board.cells[r + dr][c + dc]
                    if not cell.occupied:
                        all_occupied = False
                        break
                    if cell.broken:
                        all_unbroken = False
                        break
                    colors.add(cell.color)
                if not all_occupied or not all_unbroken:
                    break

            if all_occupied and all_unbroken:
                is_single_color = len(colors) == 1
                results.append((r, c, is_single_color))

    return results


def detect_squares(board: Board) -> list[tuple[int, int, SquareType]]:
    """Detect gold and silver squares on the board.

    Simplified detection: checks that all 16 cells in a 4x4 region
    are occupied and unbroken. Full validation of connection bits
    (ensuring exactly 4 intact tetrominoes) is deferred until the
    memory layout is fully understood.

    Returns list of (top_row, left_col, square_type).
    """
    regions = detect_4x4_regions(board)
    squares = []

    # Track which cells have been claimed by a square
    claimed = np.zeros((board.HEIGHT, board.WIDTH), dtype=bool)

    # Pass 1: Gold squares (single color)
    for r, c, is_single_color in regions:
        if not is_single_color:
            continue
        # Check no cells already claimed
        region_slice = claimed[r : r + 4, c : c + 4]
        if region_slice.any():
            continue
        squares.append((r, c, SquareType.GOLD))
        claimed[r : r + 4, c : c + 4] = True

    # Pass 2: Silver squares (multiple colors, at least 4 tetrominoes)
    for r, c, is_single_color in regions:
        if is_single_color:
            continue
        region_slice = claimed[r : r + 4, c : c + 4]
        if region_slice.any():
            continue
        squares.append((r, c, SquareType.SILVER))
        claimed[r : r + 4, c : c + 4] = True

    return squares


def count_square_potential(board: Board) -> float:
    """Score how close the board is to forming 4x4 squares.

    Checks each possible 4x4 region and scores based on how many
    of the 16 cells are filled with unbroken pieces.
    Higher scores mean the board is closer to forming squares.
    """
    potential = 0.0
    for r in range(board.HEIGHT - 3):
        for c in range(board.WIDTH - 3):
            filled = 0
            broken_count = 0
            for dr in range(4):
                for dc in range(4):
                    cell = board.cells[r + dr][c + dc]
                    if cell.occupied:
                        filled += 1
                        if cell.broken:
                            broken_count += 1

            # Only count if mostly filled and not too many broken
            if filled >= 12 and broken_count <= 2:
                potential += (filled - 11) * 0.25
    return potential
