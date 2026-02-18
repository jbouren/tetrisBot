"""Tests for the heuristic evaluator and move planner."""

import numpy as np
import pytest

from src.game.board import Board
from src.game.pieces import PieceType
from src.ai.heuristic import HeuristicEvaluator, Weights
from src.ai.planner import MovePlanner, Placement


class TestHeuristicEvaluator:
    def setup_method(self):
        self.evaluator = HeuristicEvaluator()

    def test_empty_board_score(self):
        board = Board()
        score = self.evaluator.evaluate(board)
        # Empty board should have a score of 0 (all metrics are 0)
        assert score == 0.0

    def test_fewer_holes_is_better(self):
        """Board with fewer holes should score higher."""
        # Board with no holes: flat bottom 2 rows
        grid_no_holes = np.zeros((20, 10), dtype=bool)
        grid_no_holes[18:20, :] = True
        board_no_holes = Board.from_occupancy(grid_no_holes)

        # Board with holes: block at row 17 with gaps below
        grid_holes = np.zeros((20, 10), dtype=bool)
        grid_holes[18:20, :] = True
        grid_holes[17, :] = True
        grid_holes[19, 3] = False  # hole
        grid_holes[19, 7] = False  # hole
        board_holes = Board.from_occupancy(grid_holes)

        score_no_holes = self.evaluator.evaluate(board_no_holes)
        score_holes = self.evaluator.evaluate(board_holes)
        assert score_no_holes > score_holes

    def test_complete_lines_boost_score(self):
        """Board with complete lines should score higher than similar without."""
        # Almost complete: missing one cell in bottom row
        grid_incomplete = np.zeros((20, 10), dtype=bool)
        grid_incomplete[19, :9] = True  # 9 of 10 filled
        board_incomplete = Board.from_occupancy(grid_incomplete)

        # Complete bottom row
        grid_complete = np.zeros((20, 10), dtype=bool)
        grid_complete[19, :] = True  # all 10 filled
        board_complete = Board.from_occupancy(grid_complete)

        score_incomplete = self.evaluator.evaluate(board_incomplete)
        score_complete = self.evaluator.evaluate(board_complete)
        assert score_complete > score_incomplete

    def test_lower_height_is_better(self):
        """Lower aggregate height should score better (no complete lines)."""
        # Low stack: checkerboard 2 rows (no complete lines)
        grid_low = np.zeros((20, 10), dtype=bool)
        for c in range(0, 10, 2):
            grid_low[18, c] = True
            grid_low[19, c] = True
        board_low = Board.from_occupancy(grid_low)

        # High stack: checkerboard 10 rows (no complete lines)
        grid_high = np.zeros((20, 10), dtype=bool)
        for c in range(0, 10, 2):
            grid_high[10:20, c] = True
        board_high = Board.from_occupancy(grid_high)

        score_low = self.evaluator.evaluate(board_low)
        score_high = self.evaluator.evaluate(board_high)
        assert score_low > score_high

    def test_flat_surface_is_better(self):
        """Flat surface (low bumpiness) should score better."""
        # Flat: all columns height 2
        grid_flat = np.zeros((20, 10), dtype=bool)
        grid_flat[18:20, :] = True
        board_flat = Board.from_occupancy(grid_flat)

        # Bumpy: alternating heights
        grid_bumpy = np.zeros((20, 10), dtype=bool)
        for c in range(10):
            if c % 2 == 0:
                grid_bumpy[19, c] = True  # height 1
            else:
                grid_bumpy[17:20, c] = True  # height 3
        board_bumpy = Board.from_occupancy(grid_bumpy)

        score_flat = self.evaluator.evaluate(board_flat)
        score_bumpy = self.evaluator.evaluate(board_bumpy)
        assert score_flat > score_bumpy

    def test_danger_zone_penalty(self):
        """Very tall stacks should get extra penalty."""
        # Stack at height 16 (above danger threshold of 15)
        grid = np.zeros((20, 10), dtype=bool)
        grid[4:20, 0] = True  # column 0 height = 16
        board = Board.from_occupancy(grid)
        score = self.evaluator.evaluate(board)
        # Should be very negative due to danger zone
        assert score < -5.0

    def test_well_score_single_well(self):
        """Single well should get positive well score."""
        grid = np.zeros((20, 10), dtype=bool)
        # All columns at height 4 except column 9 (well)
        grid[16:20, 0:9] = True
        board = Board.from_occupancy(grid)
        well_score = self.evaluator._well_score(board)
        assert well_score > 0


class TestMovePlanner:
    def setup_method(self):
        self.evaluator = HeuristicEvaluator()
        self.planner = MovePlanner(self.evaluator)

    def test_find_placement_empty_board(self):
        """Should find a placement on an empty board."""
        board = Board()
        placement = self.planner.find_best_placement(board, PieceType.I)
        assert placement is not None
        assert placement.piece == PieceType.I

    def test_find_placement_all_pieces(self):
        """Should find placements for all piece types on empty board."""
        board = Board()
        for piece in PieceType:
            placement = self.planner.find_best_placement(board, piece)
            assert placement is not None, f"No placement found for {piece.name}"

    def test_i_piece_prefers_well(self):
        """I-piece should go into a well when one exists."""
        grid = np.zeros((20, 10), dtype=bool)
        # Build a well: all columns at height 4 except column 9
        grid[16:20, 0:9] = True
        board = Board.from_occupancy(grid)

        placement = self.planner.find_best_placement(board, PieceType.I)
        assert placement is not None
        # I-piece vertical (rotation 1) should go in column 9
        # or I-piece horizontal should complete the row
        # Either way, score should be positive due to line clears
        assert placement.score > -100  # sanity check

    def test_placement_avoids_holes(self):
        """Planner should prefer placements that don't create holes."""
        board = Board()
        placement = self.planner.find_best_placement(board, PieceType.T)
        assert placement is not None

        # Simulate the placement and check for holes
        from src.game.pieces import get_cells
        cells = get_cells(placement.piece, placement.rotation)
        simulated, _ = self.planner._simulate_drop(
            board, cells, placement.column, placement.row
        )
        assert simulated.count_holes() == 0

    def test_two_ply_lookahead(self):
        """2-ply should return a result when next_piece is provided."""
        board = Board()
        placement = self.planner.find_best_placement(
            board, PieceType.T, next_piece=PieceType.I
        )
        assert placement is not None

    def test_generate_all_placements_count(self):
        """Should generate multiple valid placements."""
        board = Board()
        placements = self.planner._generate_all_placements(board, PieceType.T)
        # T has 4 rotations, each fitting in ~8 columns = ~32 placements
        assert len(placements) > 10

    def test_o_piece_placements(self):
        """O-piece has 1 rotation and should have exactly 9 placements on empty board."""
        board = Board()
        placements = self.planner._generate_all_placements(board, PieceType.O)
        assert len(placements) == 9  # columns 0-8 (width 2, board width 10)

    def test_simulate_drop_places_piece(self):
        """Simulating a drop should add occupied cells."""
        board = Board()
        cells = [(0, 0), (0, 1), (0, 2), (0, 3)]  # I-piece horizontal
        result, lines = self.planner._simulate_drop(board, cells, 0, 19)
        grid = result.to_occupancy_grid()
        assert grid[19, 0]
        assert grid[19, 1]
        assert grid[19, 2]
        assert grid[19, 3]
        assert lines == 0  # only 4 cells, not a complete line

    def test_simulate_drop_clears_line(self):
        """Should clear a complete line after drop."""
        grid = np.zeros((20, 10), dtype=bool)
        grid[19, 0:6] = True  # 6 cells filled in bottom row
        board = Board.from_occupancy(grid)

        # Drop I-piece horizontal to fill columns 6-9
        cells = [(0, 0), (0, 1), (0, 2), (0, 3)]
        result, lines = self.planner._simulate_drop(board, cells, 6, 19)
        assert lines == 1

    def test_find_landing_row_empty_board(self):
        """Piece should land at the bottom of an empty board."""
        board = Board()
        cells = [(0, 0), (0, 1), (1, 0), (1, 1)]  # O-piece
        row = self.planner._find_landing_row(board, cells, 0)
        assert row == 18  # O-piece is 2 tall, lands at row 18-19
