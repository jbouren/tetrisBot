"""Tests for the monoblock heuristic scoring module."""

import numpy as np
import pytest

from src.game.pieces import PieceType, get_cells
from src.game.tetris_sim import BOARD_COLS, BOARD_ROWS
from src.ai.monoblock import (
    EMPTY, UNKNOWN,
    make_color_board, simulate_drop_color, monoblock_score,
)
from src.ai.board_evaluator import _simulate_drop


class TestMakeColorBoard:
    def test_shape(self):
        board = make_color_board()
        assert board.shape == (BOARD_ROWS, BOARD_COLS)

    def test_all_empty(self):
        board = make_color_board()
        assert (board == EMPTY).all()

    def test_dtype(self):
        board = make_color_board()
        assert board.dtype == np.int8


class TestSimulateDropColor:
    def test_drop_on_empty_board(self):
        """Dropping a piece on empty board should land at bottom."""
        cb = make_color_board()
        result = simulate_drop_color(cb, PieceType.O, 0, 4)
        assert result is not None
        new_board, lines = result
        assert lines == 0
        # O piece at col 4 should land at rows 18-19, cols 4-5
        assert new_board[18, 4] == PieceType.O.value
        assert new_board[18, 5] == PieceType.O.value
        assert new_board[19, 4] == PieceType.O.value
        assert new_board[19, 5] == PieceType.O.value

    def test_drop_out_of_bounds(self):
        """Dropping piece outside board should return None."""
        cb = make_color_board()
        result = simulate_drop_color(cb, PieceType.I, 0, -1)
        assert result is None

    def test_landing_matches_binary(self):
        """Color drop should land at the same row as binary drop."""
        binary_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
        # Add some blocks at bottom
        binary_board[19, :5] = True
        binary_board[18, :3] = True

        cb = make_color_board()
        cb[binary_board] = UNKNOWN

        for piece in PieceType:
            for rot in range(4):
                cells = get_cells(piece, rot)
                min_col = min(c for _, c in cells)
                max_col = max(c for _, c in cells)
                for col in range(-min_col, BOARD_COLS - max_col):
                    bin_result = _simulate_drop(binary_board, cells, col)
                    color_result = simulate_drop_color(cb, piece, rot, col)

                    if bin_result is None:
                        assert color_result is None, (
                            f"Binary=None but color={color_result} for "
                            f"{piece.name} rot={rot} col={col}")
                        continue

                    assert color_result is not None, (
                        f"Binary succeeded but color=None for "
                        f"{piece.name} rot={rot} col={col}")

                    bin_board, _ = bin_result
                    color_board_result, _ = color_result

                    # Occupied cells should match
                    color_occupied = color_board_result != EMPTY
                    np.testing.assert_array_equal(
                        bin_board, color_occupied,
                        err_msg=f"Occupancy mismatch for {piece.name} rot={rot} col={col}"
                    )

    def test_line_clear(self):
        """Filling a row should clear it."""
        cb = make_color_board()
        # Fill row 19 except cols 0-3
        for c in range(4, BOARD_COLS):
            cb[19, c] = np.int8(PieceType.T.value)

        # Drop I piece (horizontal) at col 0 to complete the row
        result = simulate_drop_color(cb, PieceType.I, 0, 0)
        assert result is not None
        new_board, lines = result
        assert lines == 1
        # Row 19 should now be empty (cleared and shifted)
        assert (new_board[19] == EMPTY).all()

    def test_stacking(self):
        """Dropping two pieces should stack correctly."""
        cb = make_color_board()
        result = simulate_drop_color(cb, PieceType.O, 0, 0)
        assert result is not None
        cb = result[0]

        result2 = simulate_drop_color(cb, PieceType.O, 0, 0)
        assert result2 is not None
        cb = result2[0]

        # First O at rows 18-19, second at rows 16-17
        assert cb[18, 0] == PieceType.O.value
        assert cb[16, 0] == PieceType.O.value

    def test_unknown_cells_block(self):
        """UNKNOWN cells should block falling pieces (count as occupied)."""
        cb = make_color_board()
        # Partial row of unknowns (won't trigger line clear)
        cb[19, :5] = UNKNOWN

        result = simulate_drop_color(cb, PieceType.O, 0, 4)
        assert result is not None
        new_board, lines = result
        assert lines == 0
        # O piece at col 4 should land on top of unknown at (19,4)
        # Landing row = 17, so O occupies rows 17-18, cols 4-5
        assert new_board[17, 4] == PieceType.O.value
        assert new_board[17, 5] == PieceType.O.value
        assert new_board[18, 4] == PieceType.O.value
        assert new_board[18, 5] == PieceType.O.value
        # Original unknowns still there
        assert new_board[19, 0] == UNKNOWN


class TestMonoblockScore:
    def test_empty_board_zero(self):
        """Empty board should score 0."""
        cb = make_color_board()
        assert monoblock_score(cb) == 0.0

    def test_unknown_board_zero(self):
        """Board full of UNKNOWN cells should score 0 (unknown doesn't count)."""
        cb = make_color_board()
        cb[:] = UNKNOWN
        assert monoblock_score(cb) == 0.0

    def test_perfect_monoblock(self):
        """A perfect 4x4 same-color block should score 1.0 for that window."""
        cb = make_color_board()
        cb[16:20, 0:4] = np.int8(PieceType.T.value)
        score = monoblock_score(cb)
        # The 4x4 block at (16,0) contributes exactly 1.0
        # No other windows have enough cells to contribute
        assert score >= 1.0

    def test_half_filled_window(self):
        """8/16 same-type cells should score (8/16)^2 = 0.25 for that window."""
        cb = make_color_board()
        # Fill bottom-left 4x2 with T pieces (8 cells out of a 4x4 window)
        cb[16:20, 0:2] = np.int8(PieceType.T.value)
        score = monoblock_score(cb)
        # The window at (16,0) has 8/16 = 0.25, might overlap with other windows
        assert score >= 0.25

    def test_mixed_types_lower_than_mono(self):
        """Mixed types in a 4x4 region should score less than monoblock."""
        cb_mono = make_color_board()
        cb_mono[16:20, 0:4] = np.int8(PieceType.T.value)
        mono_score = monoblock_score(cb_mono)

        cb_mixed = make_color_board()
        cb_mixed[16:20, 0:2] = np.int8(PieceType.T.value)
        cb_mixed[16:20, 2:4] = np.int8(PieceType.I.value)
        mixed_score = monoblock_score(cb_mixed)

        assert mono_score > mixed_score

    def test_score_increases_with_concentration(self):
        """More same-type cells in a window should increase the score."""
        scores = []
        for n_cols in range(1, 5):
            cb = make_color_board()
            cb[16:20, 0:n_cols] = np.int8(PieceType.S.value)
            scores.append(monoblock_score(cb))

        # Score should be monotonically increasing
        for i in range(len(scores) - 1):
            assert scores[i+1] >= scores[i], \
                f"Score should increase: {scores}"
