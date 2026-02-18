"""Monoblock heuristic scoring for The New Tetris.

TNT awards bonus points for 4x4 filled regions made of intact tetrominoes:
- Monoblock: all 4 pieces are the same type (huge bonus)
- Multiblock: mixed types (smaller bonus)

This module provides a heuristic that nudges piece placement toward clustering
same-type pieces in 4x4 regions, combined with the CNN's existing evaluation.

The CNN input is binary (occupied/empty) and knows nothing about piece colors,
so this heuristic adds color-awareness on top.
"""

import numpy as np

from ..game.pieces import PieceType, ROTATION_COUNT, get_cells
from ..game.tetris_sim import BOARD_COLS, BOARD_ROWS

EMPTY = np.int8(-1)
UNKNOWN = np.int8(-2)


def make_color_board() -> np.ndarray:
    """Create an empty 20x10 color board (all cells EMPTY)."""
    board = np.full((BOARD_ROWS, BOARD_COLS), EMPTY, dtype=np.int8)
    return board


def simulate_drop_color(color_board: np.ndarray, piece_type: PieceType,
                        rotation: int, column: int
                        ) -> tuple[np.ndarray, int] | None:
    """Drop a piece on the color board, return (new_board, lines_cleared) or None.

    Uses the same landing logic as board_evaluator._simulate_drop but operates
    on the int8 color board where non-EMPTY means occupied.
    """
    cells = get_cells(piece_type, rotation)

    # Validate column bounds
    for _, dc in cells:
        c = column + dc
        if c < 0 or c >= BOARD_COLS:
            return None

    # Build occupancy mask (anything != EMPTY is occupied)
    occupied = color_board != EMPTY

    # Find landing row
    landing_row = None
    for start_row in range(BOARD_ROWS + 1):
        collision = False
        for dr, dc in cells:
            r = start_row + dr
            c = column + dc
            if r >= BOARD_ROWS:
                collision = True
                break
            if r >= 0 and occupied[r, c]:
                collision = True
                break
        if collision:
            landing_row = start_row - 1 if start_row > 0 else None
            break
    else:
        max_dr = max(dr for dr, _ in cells)
        landing_row = BOARD_ROWS - 1 - max_dr

    if landing_row is None:
        return None

    # Check placement validity
    for dr, dc in cells:
        if landing_row + dr < 0:
            return None

    # Place piece
    new_board = color_board.copy()
    for dr, dc in cells:
        new_board[landing_row + dr, column + dc] = np.int8(piece_type.value)

    # Clear lines
    full_rows = np.where(np.all(new_board != EMPTY, axis=1))[0]
    lines_cleared = len(full_rows)
    if lines_cleared > 0:
        keep = ~np.all(new_board != EMPTY, axis=1)
        remaining = new_board[keep]
        new_board = np.vstack([
            np.full((lines_cleared, BOARD_COLS), EMPTY, dtype=np.int8),
            remaining,
        ])

    return new_board, lines_cleared


def monoblock_score(color_board: np.ndarray) -> float:
    """Score a color board for monoblock potential (fast vectorized version).

    Slides a 4x4 window across all 119 positions (17 rows x 7 cols).
    For each window, finds the dominant piece type and scores by how
    concentrated that type is: (dominant_count / 16) ^ 2.

    Returns the sum of all window scores.
    """
    return _monoblock_score_fast(color_board)


def _monoblock_score_fast(color_board: np.ndarray) -> float:
    """Vectorized monoblock scoring — ~20x faster than pure Python loop."""
    # Build count arrays for each piece type across all 4x4 windows
    # There are 17 * 7 = 119 windows
    score = 0.0
    for piece_id in range(7):
        # Binary mask: where this piece type is
        mask = (color_board == piece_id).astype(np.float32)
        # Sum over 4x4 windows using cumulative sums (2D)
        # Horizontal cumsum
        cs = np.cumsum(mask, axis=1)
        h4 = cs[:, 3:] - np.concatenate([np.zeros((BOARD_ROWS, 1), dtype=np.float32), cs[:, :-4]], axis=1)
        # Vertical cumsum of h4
        cs2 = np.cumsum(h4, axis=0)
        v4 = cs2[3:, :] - np.concatenate([np.zeros((1, h4.shape[1]), dtype=np.float32), cs2[:-4, :]], axis=0)
        # v4[r, c] = count of piece_id in window (r:r+4, c:c+4) — but offset
        # We need max count per window; accumulate per-type
        if piece_id == 0:
            max_counts = v4.copy()
        else:
            max_counts = np.maximum(max_counts, v4)

    # Also need: total occupied cells per window (for the >= 4 filter)
    occ = (color_board >= 0).astype(np.float32)
    cs = np.cumsum(occ, axis=1)
    h4 = cs[:, 3:] - np.concatenate([np.zeros((BOARD_ROWS, 1), dtype=np.float32), cs[:, :-4]], axis=1)
    cs2 = np.cumsum(h4, axis=0)
    occ_counts = cs2[3:, :] - np.concatenate([np.zeros((1, h4.shape[1]), dtype=np.float32), cs2[:-4, :]], axis=0)

    # Filter: only score windows with >= 4 occupied cells
    valid = occ_counts >= 4
    scores = (max_counts[valid] / 16.0) ** 2
    return float(scores.sum())


def color_adjacency_bonus(color_board_before: np.ndarray, color_board_after: np.ndarray) -> int:
    """Count new same-type adjacent pairs created by a placement.

    Checks horizontal and vertical neighbors. Returns the increase in
    same-type adjacency count from before to after placement.
    """
    def _count_adj(board):
        occupied = board >= 0  # EMPTY = -1
        h_match = (board[:, :-1] == board[:, 1:]) & occupied[:, :-1] & occupied[:, 1:]
        v_match = (board[:-1, :] == board[1:, :]) & occupied[:-1, :] & occupied[1:, :]
        return int(h_match.sum()) + int(v_match.sum())
    return _count_adj(color_board_after) - _count_adj(color_board_before)


def find_best_placement_combined(
    evaluator,  # BoardEvaluator
    binary_board: np.ndarray,
    color_board: np.ndarray,
    piece_type: PieceType,
    weight: float,
) -> tuple[int, int, float, float, float]:
    """Find best placement using CNN + monoblock heuristic.

    Returns (rotation, column, combined_score, cnn_score, mono_score).
    """
    from .board_evaluator import _simulate_drop

    candidates = []
    result_boards = []      # binary boards for CNN
    color_boards = []       # color boards for monoblock

    num_rots = ROTATION_COUNT[piece_type]
    for rot in range(num_rots):
        cells = get_cells(piece_type, rot)
        min_col = min(c for _, c in cells)
        max_col = max(c for _, c in cells)

        for col in range(-min_col, BOARD_COLS - max_col):
            # Binary simulation (for CNN)
            bin_result = _simulate_drop(binary_board, cells, col)
            if bin_result is None:
                continue
            bin_board, game_over = bin_result
            if game_over:
                continue

            # Color simulation (for monoblock)
            color_result = simulate_drop_color(color_board, piece_type, rot, col)
            if color_result is None:
                continue

            candidates.append((rot, col))
            result_boards.append(bin_board)
            color_boards.append(color_result[0])

    if not candidates:
        return 0, BOARD_COLS // 2, float("-inf"), float("-inf"), 0.0

    # Batch CNN scoring
    cnn_scores = evaluator.evaluate_batch(result_boards)

    # Monoblock scoring for each candidate
    mono_scores = np.array([monoblock_score(cb) for cb in color_boards])

    # Combined score
    combined = cnn_scores + weight * mono_scores

    best_idx = int(np.argmax(combined))
    rot, col = candidates[best_idx]
    return (rot, col,
            float(combined[best_idx]),
            float(cnn_scores[best_idx]),
            float(mono_scores[best_idx]))
