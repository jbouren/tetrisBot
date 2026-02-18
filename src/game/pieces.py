"""Tetris piece definitions and rotation tables.

The New Tetris uses a rotation system similar to (but not identical to)
the later SRS (Super Rotation System). The exact offsets should be verified
against the actual game behavior using the memory scanner.
"""

from enum import IntEnum


class PieceType(IntEnum):
    I = 0
    J = 1
    L = 2
    O = 3
    S = 4
    T = 5
    Z = 6


PIECE_NAMES = {
    PieceType.I: "I",
    PieceType.J: "J",
    PieceType.L: "L",
    PieceType.O: "O",
    PieceType.S: "S",
    PieceType.T: "T",
    PieceType.Z: "Z",
}

# Piece shapes: list of (row, col) offsets for each rotation state.
# row increases downward, col increases rightward.
# Rotation 0 = spawn orientation.
# These are approximate and should be verified against the actual game.
PIECE_SHAPES: dict[PieceType, list[list[tuple[int, int]]]] = {
    PieceType.I: [
        [(0, 0), (0, 1), (0, 2), (0, 3)],  # ----
        [(0, 0), (1, 0), (2, 0), (3, 0)],  # |
        [(0, 0), (0, 1), (0, 2), (0, 3)],  # ---- (same as 0)
        [(0, 0), (1, 0), (2, 0), (3, 0)],  # | (same as 1)
    ],
    PieceType.J: [
        [(0, 0), (0, 1), (0, 2), (1, 2)],  # J (now spawn)
        [(0, 1), (1, 1), (2, 0), (2, 1)],  # J rotated CW
        [(0, 0), (1, 0), (1, 1), (1, 2)],  # J rotated 180
        [(0, 0), (0, 1), (1, 0), (2, 0)],  # J rotated CCW
    ],
    PieceType.L: [
        [(0, 0), (0, 1), (0, 2), (1, 0)],  # L (now spawn)
        [(0, 0), (0, 1), (1, 1), (2, 1)],  # L rotated CW
        [(0, 2), (1, 0), (1, 1), (1, 2)],  # L rotated 180
        [(0, 0), (1, 0), (2, 0), (2, 1)],  # L rotated CCW
    ],
    PieceType.O: [
        [(0, 0), (0, 1), (1, 0), (1, 1)],  # square
        [(0, 0), (0, 1), (1, 0), (1, 1)],  # same
        [(0, 0), (0, 1), (1, 0), (1, 1)],  # same
        [(0, 0), (0, 1), (1, 0), (1, 1)],  # same
    ],
    PieceType.S: [
        [(0, 1), (0, 2), (1, 0), (1, 1)],  # S
        [(0, 0), (1, 0), (1, 1), (2, 1)],  # S rotated
        [(0, 1), (0, 2), (1, 0), (1, 1)],  # same as 0
        [(0, 0), (1, 0), (1, 1), (2, 1)],  # same as 1
    ],
    PieceType.T: [
        [(0, 1), (1, 0), (1, 1), (1, 2)],  # T pointing up
        [(0, 0), (1, 0), (1, 1), (2, 0)],  # T pointing right
        [(0, 0), (0, 1), (0, 2), (1, 1)],  # T pointing down
        [(0, 1), (1, 0), (1, 1), (2, 1)],  # T pointing left
    ],
    PieceType.Z: [
        [(0, 0), (0, 1), (1, 1), (1, 2)],  # Z
        [(0, 1), (1, 0), (1, 1), (2, 0)],  # Z rotated
        [(0, 0), (0, 1), (1, 1), (1, 2)],  # same as 0
        [(0, 1), (1, 0), (1, 1), (2, 0)],  # same as 1
    ],
}

# Number of distinct rotations per piece type.
# In The New Tetris, S and Z have 4 rotations (not 2 like standard Tetris).
ROTATION_COUNT = {
    PieceType.I: 2,
    PieceType.J: 4,
    PieceType.L: 4,
    PieceType.O: 1,
    PieceType.S: 4,
    PieceType.T: 4,
    PieceType.Z: 4,
}


def get_cells(piece: PieceType, rotation: int) -> list[tuple[int, int]]:
    """Get (row, col) offsets for a piece in a given rotation."""
    return PIECE_SHAPES[piece][rotation % 4]


_ROTATION_OFFSETS = {}

def _calculate_offsets():
    for piece in PieceType:
        cells_r0 = get_cells(piece, 0)
        min_c_r0 = min(c for _, c in cells_r0)
        _ROTATION_OFFSETS[piece] = {}
        for r in range(4):
            cells_r = get_cells(piece, r)
            min_c_r = min(c for _, c in cells_r)
            _ROTATION_OFFSETS[piece][r] = min_c_r0 - min_c_r

_calculate_offsets()

def get_col_after_rotation(piece: PieceType, rotation: int, spawn_col: int) -> int:
    """Get the column of the piece's bounding box after rotation."""
    return spawn_col + _ROTATION_OFFSETS[piece][rotation % 4]



def get_width(piece: PieceType, rotation: int) -> int:
    """Get the width (column span) of a piece in a given rotation."""
    cells = get_cells(piece, rotation)
    cols = [c for _, c in cells]
    return max(cols) - min(cols) + 1


def get_height(piece: PieceType, rotation: int) -> int:
    """Get the height (row span) of a piece in a given rotation."""
    cells = get_cells(piece, rotation)
    rows = [r for r, _ in cells]
    return max(rows) - min(rows) + 1


def normalize_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Shift cells so the minimum row and column are both 0."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)
