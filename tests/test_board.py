"""Tests for the Board class and Cell parsing."""

import numpy as np
import pytest

from src.game.board import Board, Cell


class TestCell:
    def test_empty_cell(self):
        cell = Cell(0, cell_size=2)
        assert cell.color == 0
        assert not cell.occupied
        assert not cell.conn_right
        assert not cell.broken

    def test_occupied_cell_2byte(self):
        # color=1 in bits 15-12 -> 0x1000
        cell = Cell(0x1000, cell_size=2)
        assert cell.color == 1
        assert cell.occupied

    def test_connection_bits(self):
        # color=2, conn_right=1, conn_down=1 -> 0x2C00
        raw = (2 << 12) | 0x800 | 0x400
        cell = Cell(raw, cell_size=2)
        assert cell.color == 2
        assert cell.occupied
        assert cell.conn_right
        assert cell.conn_down
        assert not cell.conn_left
        assert not cell.conn_up

    def test_broken_flag(self):
        raw = (1 << 12) | 0x80
        cell = Cell(raw, cell_size=2)
        assert cell.occupied
        assert cell.broken

    def test_1byte_cell(self):
        # color=3 in bits 7-4 -> 0x30
        cell = Cell(0x30, cell_size=1)
        assert cell.color == 3
        assert cell.occupied

    def test_1byte_empty(self):
        cell = Cell(0x00, cell_size=1)
        assert cell.color == 0
        assert not cell.occupied


class TestBoard:
    def _make_empty_board(self) -> Board:
        return Board()

    def _make_board_with_bottom_row(self) -> Board:
        """Create a board with the bottom row fully occupied."""
        grid = np.zeros((20, 10), dtype=bool)
        grid[19, :] = True  # fill bottom row
        return Board.from_occupancy(grid)

    def _make_board_with_hole(self) -> Board:
        """Board with bottom 2 rows full except one hole."""
        grid = np.zeros((20, 10), dtype=bool)
        grid[18, :] = True
        grid[19, :] = True
        grid[19, 5] = False  # hole at column 5, bottom row
        return Board.from_occupancy(grid)

    def test_empty_board_dimensions(self):
        board = self._make_empty_board()
        assert len(board.cells) == 20
        assert len(board.cells[0]) == 10

    def test_empty_board_metrics(self):
        board = self._make_empty_board()
        assert board.count_holes() == 0
        assert board.count_complete_lines() == 0
        assert board.aggregate_height() == 0
        assert board.max_height() == 0
        assert board.bumpiness() == 0

    def test_occupancy_grid_empty(self):
        board = self._make_empty_board()
        grid = board.to_occupancy_grid()
        assert grid.shape == (20, 10)
        assert not grid.any()

    def test_occupancy_grid_with_data(self):
        board = self._make_board_with_bottom_row()
        grid = board.to_occupancy_grid()
        assert grid[19].all()  # bottom row all occupied
        assert not grid[0].any()  # top row all empty

    def test_column_heights_empty(self):
        board = self._make_empty_board()
        heights = board.column_heights()
        assert (heights == 0).all()

    def test_column_heights_bottom_row(self):
        board = self._make_board_with_bottom_row()
        heights = board.column_heights()
        assert (heights == 1).all()

    def test_column_heights_varied(self):
        grid = np.zeros((20, 10), dtype=bool)
        grid[19, 0] = True   # col 0: height 1
        grid[18, 1] = True   # col 1: height 2
        grid[15, 2] = True   # col 2: height 5
        board = Board.from_occupancy(grid)
        heights = board.column_heights()
        assert heights[0] == 1
        assert heights[1] == 2
        assert heights[2] == 5
        assert heights[3] == 0

    def test_count_holes_none(self):
        board = self._make_board_with_bottom_row()
        assert board.count_holes() == 0

    def test_count_holes_one(self):
        board = self._make_board_with_hole()
        assert board.count_holes() == 1

    def test_count_holes_column(self):
        """Hole below a block in a single column."""
        grid = np.zeros((20, 10), dtype=bool)
        grid[17, 3] = True  # block at row 17
        # rows 18, 19 empty in col 3 -> 2 holes
        board = Board.from_occupancy(grid)
        assert board.count_holes() == 2

    def test_complete_lines_none(self):
        board = self._make_board_with_hole()
        assert board.count_complete_lines() == 1  # row 18 is complete

    def test_complete_lines_full_bottom(self):
        board = self._make_board_with_bottom_row()
        assert board.count_complete_lines() == 1

    def test_get_complete_lines(self):
        board = self._make_board_with_bottom_row()
        lines = board.get_complete_lines()
        assert lines == [19]

    def test_clear_lines(self):
        board = self._make_board_with_bottom_row()
        new_board, count = board.clear_lines()
        assert count == 1
        assert new_board.count_complete_lines() == 0
        # After clearing, board should be empty
        assert new_board.aggregate_height() == 0

    def test_clear_lines_no_lines(self):
        board = self._make_empty_board()
        new_board, count = board.clear_lines()
        assert count == 0

    def test_bumpiness_flat(self):
        board = self._make_board_with_bottom_row()
        assert board.bumpiness() == 0

    def test_bumpiness_uneven(self):
        grid = np.zeros((20, 10), dtype=bool)
        grid[19, 0] = True  # col 0: height 1
        grid[18, 1] = True  # col 1: height 2
        grid[19, 1] = True
        board = Board.from_occupancy(grid)
        # bumpiness = |1-2| + |2-0| + |0-0|*7 = 1 + 2 = 3
        assert board.bumpiness() == 3

    def test_ascii_empty(self):
        board = self._make_empty_board()
        ascii_repr = board.to_ascii()
        assert "+----------+" in ascii_repr
        assert "|..........|" in ascii_repr

    def test_ascii_with_blocks(self):
        board = self._make_board_with_bottom_row()
        ascii_repr = board.to_ascii()
        assert "|##########|" in ascii_repr

    def test_from_occupancy_roundtrip(self):
        grid = np.zeros((20, 10), dtype=bool)
        grid[15:20, 0:5] = True
        board = Board.from_occupancy(grid)
        result = board.to_occupancy_grid()
        np.testing.assert_array_equal(grid, result)
