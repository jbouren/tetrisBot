"""Numba-accelerated simulation for color board drop + line clearing.

Provides ~10-50x speedup over pure Python simulate_drop_color for the
training hot path (enumerate all placements per piece per step).
"""

import numpy as np
import numba
from numba import njit, int8, int32, boolean

from ..game.pieces import PieceType, PIECE_SHAPES, ROTATION_COUNT
from ..game.tetris_sim import BOARD_ROWS, BOARD_COLS

EMPTY = int8(-1)

# Pre-compute all piece cells as numpy arrays for Numba.
# PIECE_CELLS[piece_value][rotation] = (4, 2) int32 array of (dr, dc)
# We store as a flat 3D array: (7 pieces * 4 rotations, 4 cells, 2 coords)
# Index: piece_value * 4 + rotation
_ALL_CELLS = np.zeros((7 * 4, 4, 2), dtype=np.int32)
for _pt in PieceType:
    for _rot in range(4):
        _cells = PIECE_SHAPES[_pt][_rot]
        for _i, (_dr, _dc) in enumerate(_cells):
            _ALL_CELLS[_pt.value * 4 + _rot, _i, 0] = _dr
            _ALL_CELLS[_pt.value * 4 + _rot, _i, 1] = _dc

# Pre-compute valid column ranges for each (piece, rotation)
# VALID_COL_RANGE[piece*4+rot] = (min_valid_col, max_valid_col+1)
_COL_RANGES = np.zeros((7 * 4, 2), dtype=np.int32)
for _pt in PieceType:
    for _rot in range(ROTATION_COUNT[_pt]):
        _cells = PIECE_SHAPES[_pt][_rot]
        _min_c = min(c for _, c in _cells)
        _max_c = max(c for _, c in _cells)
        _idx = _pt.value * 4 + _rot
        _COL_RANGES[_idx, 0] = -_min_c
        _COL_RANGES[_idx, 1] = BOARD_COLS - _max_c

# Rotation counts as numpy array
_ROT_COUNTS = np.array([ROTATION_COUNT[pt] for pt in PieceType], dtype=np.int32)


@njit(cache=True)
def _find_landing_row(board, cells, column):
    """Find landing row for piece cells at column. Returns -999 if invalid."""
    rows = BOARD_ROWS
    # Validate column bounds
    for i in range(4):
        c = column + cells[i, 1]
        if c < 0 or c >= BOARD_COLS:
            return -999

    # Find landing row
    for start_row in range(rows + 1):
        collision = False
        for i in range(4):
            dr = cells[i, 0]
            dc = cells[i, 1]
            r = start_row + dr
            c = column + dc
            if r >= rows:
                collision = True
                break
            if r >= 0 and board[r, c] != EMPTY:
                collision = True
                break
        if collision:
            if start_row > 0:
                landing = start_row - 1
            else:
                return -999
            # Validate placement
            for i in range(4):
                if landing + cells[i, 0] < 0:
                    return -999
            return landing

    # No collision — piece falls to bottom
    max_dr = int8(0)
    for i in range(4):
        if cells[i, 0] > max_dr:
            max_dr = cells[i, 0]
    landing = rows - 1 - max_dr
    for i in range(4):
        if landing + cells[i, 0] < 0:
            return -999
    return landing


@njit(cache=True)
def simulate_drop_fast(board, piece_value, cells, column):
    """Drop piece on color board. Returns (new_board, lines_cleared, valid).

    board: (20, 10) int8, piece_value: int8, cells: (4, 2) int32, column: int32
    new_board is modified in-place only if valid=True.
    """
    landing = _find_landing_row(board, cells, column)
    if landing == -999:
        return board, 0, False

    # Place piece (copy first)
    new_board = board.copy()
    for i in range(4):
        r = landing + cells[i, 0]
        c = column + cells[i, 1]
        new_board[r, c] = piece_value

    # Clear lines
    lines_cleared = 0
    # Check which rows are full
    full = np.zeros(BOARD_ROWS, dtype=numba.boolean)
    for r in range(BOARD_ROWS):
        row_full = True
        for c in range(BOARD_COLS):
            if new_board[r, c] == EMPTY:
                row_full = False
                break
        if row_full:
            full[r] = True
            lines_cleared += 1

    if lines_cleared > 0:
        # Compact: move non-full rows down
        result = np.full((BOARD_ROWS, BOARD_COLS), EMPTY, dtype=int8)
        dest = BOARD_ROWS - 1
        for r in range(BOARD_ROWS - 1, -1, -1):
            if not full[r]:
                for c in range(BOARD_COLS):
                    result[dest, c] = new_board[r, c]
                dest -= 1
        new_board = result

    return new_board, lines_cleared, True


@njit(cache=True)
def enumerate_placements(board, piece_value, all_cells, col_ranges, num_rots):
    """Enumerate all valid placements for a piece. Returns list of (rot, col, result_board).

    More efficient than calling simulate_drop_fast in a Python loop.
    Returns: (result_boards array (N, 20, 10), rot_col array (N, 2), count)
    Max placements per piece: 4 rots * 10 cols = 40
    """
    max_placements = 40
    result_boards = np.full((max_placements, BOARD_ROWS, BOARD_COLS), EMPTY, dtype=int8)
    rot_col = np.zeros((max_placements, 2), dtype=int32)
    lines = np.zeros(max_placements, dtype=int32)
    count = 0

    for rot in range(num_rots):
        idx = piece_value * 4 + rot
        cells = all_cells[idx]
        col_lo = col_ranges[idx, 0]
        col_hi = col_ranges[idx, 1]

        for col in range(col_lo, col_hi):
            new_board, lc, valid = simulate_drop_fast(board, int8(piece_value), cells, col)
            if not valid:
                continue
            # Check game over: top 2 rows
            game_over = False
            for c in range(BOARD_COLS):
                if new_board[0, c] != EMPTY or new_board[1, c] != EMPTY:
                    game_over = True
                    break
            if game_over:
                continue

            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    result_boards[count, r, c] = new_board[r, c]
            rot_col[count, 0] = rot
            rot_col[count, 1] = col
            lines[count] = lc
            count += 1

    return result_boards, rot_col, lines, count


def enumerate_placements_py(board, piece_type):
    """Python wrapper for enumerate_placements. Returns (boards, rot_cols, lines, count).

    board: (20, 10) int8 color board
    piece_type: PieceType enum
    """
    pv = piece_type.value
    nr = _ROT_COUNTS[pv]
    return enumerate_placements(board, pv, _ALL_CELLS, _COL_RANGES, nr)


def warmup():
    """Call once to trigger Numba JIT compilation."""
    board = np.full((BOARD_ROWS, BOARD_COLS), EMPTY, dtype=np.int8)
    enumerate_placements(board, 0, _ALL_CELLS, _COL_RANGES, 2)
