"""Pure Python Tetris simulator for fast training.

No emulator, no CV, no screen capture. Just Python + numpy.
Designed for thousands of games per second for RL training.

Implements The New Tetris square mechanic:
- A 4x4 filled region made of exactly 4 intact tetrominoes becomes a "square"
- Monosquare: all 4 tetrominoes are the same type (higher value)
- Multisquare: 4 tetrominoes of mixed types
- Pieces broken by line clears cannot form squares
- Cells already part of a square cannot form another
"""

import random

import numpy as np

from .pieces import PieceType, PIECE_SHAPES, ROTATION_COUNT, get_cells, get_width


BOARD_ROWS = 20
BOARD_COLS = 10

# Scoring: lines cleared -> reward
LINE_REWARDS = {0: 0, 1: 1, 2: 3, 3: 5, 4: 8}

# Square bonuses
MONOSQUARE_REWARD = 10
MULTISQUARE_REWARD = 5

GAME_OVER_PENALTY = -5


class TetrisSim:
    """Fast standalone Tetris engine for training.

    Board is a 20x10 numpy bool array. Pieces use definitions from pieces.py.
    Uses standard 7-bag randomizer. Tracks The New Tetris square mechanic.

    Usage:
        sim = TetrisSim()
        obs = sim.reset()
        while not sim.game_over:
            obs, reward, done, info = sim.step(rotation=0, column=3)
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.board: np.ndarray = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
        self.current_piece: PieceType = PieceType.I
        self.next_pieces: list[PieceType] = []
        self.score: int = 0
        self.lines_cleared: int = 0
        self.pieces_placed: int = 0
        self.game_over: bool = False
        self._bag: list[PieceType] = []

        # Square tracking: which piece placed each cell, and piece metadata
        self._cell_piece_id: np.ndarray = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int32)
        self._squared: np.ndarray = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
        self._piece_type: dict[int, PieceType] = {}   # piece_id -> PieceType
        self._piece_intact: dict[int, bool] = {}       # piece_id -> still whole?
        self._next_piece_id: int = 1
        self.mono_squares: int = 0
        self.multi_squares: int = 0

    def reset(self) -> dict:
        """Reset the game and return the initial observation."""
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self._bag = []

        self._cell_piece_id = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int32)
        self._squared = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
        self._piece_type = {}
        self._piece_intact = {}
        self._next_piece_id = 1
        self.mono_squares = 0
        self.multi_squares = 0

        # Fill the queue: current + 3 next
        self.current_piece = self._draw_piece()
        self.next_pieces = [self._draw_piece() for _ in range(3)]

        return self._observation()

    def step(self, rotation: int, column: int) -> tuple[dict, float, bool, dict]:
        """Place the current piece at (rotation, column) and advance.

        Args:
            rotation: Rotation state (0 to ROTATION_COUNT-1).
            column: Leftmost column for piece placement.

        Returns:
            (observation, reward, done, info)
        """
        if self.game_over:
            return self._observation(), 0.0, True, {"reason": "already_over"}

        # Clamp rotation
        rotation = rotation % ROTATION_COUNT[self.current_piece]

        # Try to place the piece
        lines, valid = self._place_piece(self.current_piece, rotation, column)

        if not valid:
            self.game_over = True
            return self._observation(), GAME_OVER_PENALTY, True, {"reason": "invalid_placement"}

        self.pieces_placed += 1
        self.lines_cleared += lines
        reward = float(LINE_REWARDS.get(lines, lines * 2))

        # Check for new squares after placement + line clear
        new_mono, new_multi = self._detect_squares()
        self.mono_squares += new_mono
        self.multi_squares += new_multi
        reward += new_mono * MONOSQUARE_REWARD + new_multi * MULTISQUARE_REWARD

        self.score += int(reward)

        # Advance to next piece
        self.current_piece = self.next_pieces.pop(0)
        self.next_pieces.append(self._draw_piece())

        # Check game over: top 2 rows occupied
        if self._check_game_over():
            self.game_over = True
            reward += GAME_OVER_PENALTY
            return self._observation(), reward, True, {"reason": "topped_out"}

        info = {"lines": lines, "new_mono_squares": new_mono, "new_multi_squares": new_multi}
        return self._observation(), reward, False, info

    def _draw_piece(self) -> PieceType:
        """Draw next piece from the 7-bag randomizer."""
        if not self._bag:
            self._bag = list(PieceType)
            self._rng.shuffle(self._bag)
        return self._bag.pop()

    def _place_piece(self, piece: PieceType, rotation: int, column: int) -> tuple[int, bool]:
        """Drop piece at column and place it. Returns (lines_cleared, valid)."""
        cells = get_cells(piece, rotation)

        # Validate column bounds
        for _, dc in cells:
            c = column + dc
            if c < 0 or c >= BOARD_COLS:
                return 0, False

        # Find landing row
        landing_row = self._find_landing_row(cells, column)
        if landing_row is None:
            return 0, False

        # Check if placement is above the board (game over)
        for dr, dc in cells:
            r = landing_row + dr
            if r < 0:
                return 0, False

        # Assign a piece ID and place
        pid = self._next_piece_id
        self._next_piece_id += 1
        self._piece_type[pid] = piece
        self._piece_intact[pid] = True

        placed_rows = []
        placed_cols = []
        for dr, dc in cells:
            r = landing_row + dr
            c = column + dc
            self.board[r, c] = True
            self._cell_piece_id[r, c] = pid
            placed_rows.append(r)
            placed_cols.append(c)

        # Track placement bounds for efficient square scanning
        self._last_place_bounds = (
            min(placed_rows), max(placed_rows),
            min(placed_cols), max(placed_cols),
        )

        # Clear lines
        lines = self._clear_lines()
        return lines, True

    def _find_landing_row(self, cells: list[tuple[int, int]], column: int) -> int | None:
        """Find the row where a piece lands when dropped at column.

        Returns the row offset for the piece origin, or None if it can't fit.
        """
        for start_row in range(BOARD_ROWS + 1):
            for dr, dc in cells:
                r = start_row + dr
                c = column + dc
                if r >= BOARD_ROWS:
                    return start_row - 1 if start_row > 0 else None
                if r >= 0 and self.board[r, c]:
                    return start_row - 1 if start_row > 0 else None
        # Piece reached the bottom
        max_dr = max(dr for dr, _ in cells)
        return BOARD_ROWS - 1 - max_dr

    def _clear_lines(self) -> int:
        """Remove complete lines and shift down. Returns count cleared.

        Also marks any pieces that lose cells as broken (can't form squares).
        """
        full_rows = np.where(self.board.all(axis=1))[0]
        if len(full_rows) == 0:
            return 0

        # Mark pieces in cleared rows as broken
        for r in full_rows:
            for c in range(BOARD_COLS):
                pid = self._cell_piece_id[r, c]
                if pid > 0:
                    self._piece_intact[pid] = False

        # Keep non-full rows
        keep_mask = ~self.board.all(axis=1)
        remaining = self.board[keep_mask]
        remaining_ids = self._cell_piece_id[keep_mask]
        remaining_sq = self._squared[keep_mask]

        # Stack empty rows on top
        n_cleared = BOARD_ROWS - remaining.shape[0]
        empty_board = np.zeros((n_cleared, BOARD_COLS), dtype=bool)
        empty_ids = np.zeros((n_cleared, BOARD_COLS), dtype=np.int32)
        empty_sq = np.zeros((n_cleared, BOARD_COLS), dtype=bool)

        self.board = np.vstack([empty_board, remaining])
        self._cell_piece_id = np.vstack([empty_ids, remaining_ids])
        self._squared = np.vstack([empty_sq, remaining_sq])

        return n_cleared

    def _detect_squares(self) -> tuple[int, int]:
        """Scan for new 4x4 squares near the last placed piece.

        A valid square is a fully filled 4x4 region containing exactly 4 intact
        tetrominoes, with no cells already part of another square.

        Only checks 4x4 regions that overlap the last placed piece (a new square
        can only form if it includes cells from the piece just placed).

        Returns (mono_count, multi_count).
        """
        mono = 0
        multi = 0

        if not hasattr(self, '_last_place_bounds'):
            return 0, 0

        min_r, max_r, min_c, max_c = self._last_place_bounds

        # A 4x4 region at (r, c) overlaps the placed piece if its rows [r, r+3]
        # intersect [min_r, max_r] and its cols [c, c+3] intersect [min_c, max_c].
        r_lo = max(0, min_r - 3)
        r_hi = min(BOARD_ROWS - 4, max_r)
        c_lo = max(0, min_c - 3)
        c_hi = min(BOARD_COLS - 4, max_c)

        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                # Quick check: is the 4x4 region fully occupied?
                if not self.board[r:r+4, c:c+4].all():
                    continue

                # No cells already squared
                if self._squared[r:r+4, c:c+4].any():
                    continue

                # Collect piece IDs in the region
                id_region = self._cell_piece_id[r:r+4, c:c+4]
                piece_ids = set(id_region.flat)
                piece_ids.discard(0)

                # Must be exactly 4 tetrominoes (4 cells each = 16 total)
                if len(piece_ids) != 4:
                    continue

                # All 4 pieces must be intact (not broken by line clears)
                if not all(self._piece_intact.get(pid, False) for pid in piece_ids):
                    continue

                # Each piece must contribute exactly 4 cells to the region
                valid = True
                for pid in piece_ids:
                    if int((id_region == pid).sum()) != 4:
                        valid = False
                        break
                if not valid:
                    continue

                # It's a square! Mark cells and classify
                self._squared[r:r+4, c:c+4] = True

                types = {self._piece_type[pid] for pid in piece_ids}
                if len(types) == 1:
                    mono += 1
                else:
                    multi += 1

        return mono, multi

    def _check_game_over(self) -> bool:
        """Game over if any of the top 2 rows have blocks."""
        return bool(self.board[:2].any())

    def _observation(self) -> dict:
        """Return the current observation."""
        return {
            "board": self.board.copy(),
            "current_piece": self.current_piece,
            "next_pieces": list(self.next_pieces),
        }

    def get_color_board(self) -> np.ndarray:
        """Return 20x10 int8 color board (EMPTY=-1, piece values 0-6)."""
        color_board = np.full((BOARD_ROWS, BOARD_COLS), np.int8(-1), dtype=np.int8)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                pid = self._cell_piece_id[r, c]
                if pid > 0 and pid in self._piece_type:
                    color_board[r, c] = np.int8(self._piece_type[pid].value)
        return color_board

    def get_valid_actions(self) -> list[tuple[int, int]]:
        """Return all valid (rotation, column) pairs for the current piece."""
        actions = []
        piece = self.current_piece
        num_rots = ROTATION_COUNT[piece]

        for rot in range(num_rots):
            cells = get_cells(piece, rot)
            min_col = min(c for _, c in cells)
            max_col = max(c for _, c in cells)

            for col in range(-min_col, BOARD_COLS - max_col):
                actions.append((rot, col))

        return actions

    def render(self) -> str:
        """ASCII rendering of the current board state.

        '#' = normal block, 'S' = part of a formed square, '.' = empty.
        """
        lines = []
        lines.append(f"Piece: {self.current_piece.name}  "
                     f"Next: {','.join(p.name for p in self.next_pieces)}  "
                     f"Score: {self.score}  Lines: {self.lines_cleared}  "
                     f"Placed: {self.pieces_placed}  "
                     f"Squares: {self.mono_squares}M+{self.multi_squares}X")
        lines.append("+" + "-" * BOARD_COLS + "+")
        for row in range(BOARD_ROWS):
            row_str = ""
            for col in range(BOARD_COLS):
                if self._squared[row, col]:
                    row_str += "S"
                elif self.board[row, col]:
                    row_str += "#"
                else:
                    row_str += "."
            lines.append("|" + row_str + "|")
        lines.append("+" + "-" * BOARD_COLS + "+")
        return "\n".join(lines)
