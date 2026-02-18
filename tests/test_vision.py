"""Tests for the computer vision module."""

import numpy as np
import pytest

from src.game.pieces import PieceType
from src.game.vision import (
    BOARD_ROWS, BOARD_COLS, BOARD_LEFT, BOARD_TOP, BOARD_RIGHT, BOARD_BOTTOM,
    CELL_WIDTH, CELL_HEIGHT, OCCUPIED_THRESHOLD,
    PREVIEW_PIECE_X0, PREVIEW_PIECE_X1, PREVIEW_SLOTS,
    PIECE_COLORS,
    get_cell_center, sample_cell, read_board, read_board_colors,
    read_board_brightness, read_board_clean, strip_ghost,
    board_to_ascii, classify_color, sample_preview_slot, read_preview,
    detect_falling_piece,
)


class TestBoardReading:
    def _make_frame(self, cells=None, brightness=200):
        """Create a synthetic 480x640x3 frame with specified cells lit."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if cells:
            for col, row in cells:
                cx, cy = get_cell_center(col, row)
                # Fill a small region around the center
                x0 = max(0, int(cx - 4))
                x1 = min(640, int(cx + 4))
                y0 = max(0, int(cy - 4))
                y1 = min(480, int(cy + 4))
                frame[y0:y1, x0:x1] = brightness
        return frame

    def test_empty_board(self):
        frame = self._make_frame()
        board = read_board(frame)
        assert board.shape == (20, 10)
        assert not board.any()

    def test_single_cell(self):
        frame = self._make_frame(cells=[(5, 10)])
        board = read_board(frame)
        assert board[10, 5]
        assert board.sum() == 1

    def test_i_piece_horizontal(self):
        frame = self._make_frame(cells=[(3, 7), (4, 7), (5, 7), (6, 7)])
        board = read_board(frame)
        assert board[7, 3] and board[7, 4] and board[7, 5] and board[7, 6]
        assert board.sum() == 4

    def test_board_to_ascii(self):
        board = np.zeros((20, 10), dtype=bool)
        board[19, 0] = True
        board[19, 9] = True
        ascii_art = board_to_ascii(board)
        lines = ascii_art.split("\n")
        assert lines[0] == "+----------+"
        assert lines[-1] == "+----------+"
        assert lines[20] == "|#........#|"

    def test_read_board_colors(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Place a cyan cell at (3, 7)
        cx, cy = get_cell_center(3, 7)
        x0, x1 = max(0, int(cx - 4)), min(640, int(cx + 4))
        y0, y1 = max(0, int(cy - 4)), min(480, int(cy + 4))
        frame[y0:y1, x0:x1] = [40, 135, 190]

        occupancy, colors = read_board_colors(frame)
        assert occupancy[7, 3]
        # Color should be close to what we set
        assert colors[7, 3, 0] > 30  # Red channel
        assert colors[7, 3, 1] > 100  # Green channel


class TestColorClassification:
    def test_cyan_is_i_piece(self):
        assert classify_color(np.array([40, 135, 190])) == PieceType.I

    def test_yellow_is_t_piece(self):
        assert classify_color(np.array([186, 169, 24])) == PieceType.T

    def test_purple_is_j_piece(self):
        assert classify_color(np.array([108, 55, 168])) == PieceType.J

    def test_green_is_s_piece(self):
        assert classify_color(np.array([65, 173, 87])) == PieceType.S

    def test_red_is_z_piece(self):
        assert classify_color(np.array([168, 62, 62])) == PieceType.Z

    def test_gray_is_o_piece(self):
        assert classify_color(np.array([146, 144, 142])) == PieceType.O

    def test_near_cyan_still_i(self):
        # Slight variations should still match
        assert classify_color(np.array([50, 140, 185])) == PieceType.I

    def test_near_yellow_still_t(self):
        assert classify_color(np.array([190, 175, 30])) == PieceType.T

    def test_very_far_color_returns_none(self):
        # A color far from all references
        assert classify_color(np.array([0, 0, 0])) is None

    def test_all_piece_colors_roundtrip(self):
        """Each reference color should classify to its own piece type."""
        for piece_type, color in PIECE_COLORS.items():
            assert classify_color(color) == piece_type


class TestPreviewReading:
    def _make_preview_frame(self, slot_colors=None):
        """Create a frame with colored blocks in preview slots."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if slot_colors:
            for slot, color in slot_colors.items():
                y0, y1 = PREVIEW_SLOTS[slot]
                frame[y0:y1, PREVIEW_PIECE_X0:PREVIEW_PIECE_X1] = color
        return frame

    def test_no_preview_pieces(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        preview = read_preview(frame)
        assert preview == [None, None, None]

    def test_single_slot_cyan(self):
        frame = self._make_preview_frame({0: [40, 135, 190]})
        preview = read_preview(frame)
        assert preview[0] == PieceType.I
        assert preview[1] is None
        assert preview[2] is None

    def test_all_three_slots(self):
        frame = self._make_preview_frame({
            0: [186, 169, 24],   # T (yellow)
            1: [40, 135, 190],   # I (cyan)
            2: [65, 173, 87],    # S (green)
        })
        preview = read_preview(frame)
        assert preview[0] == PieceType.T
        assert preview[1] == PieceType.I
        assert preview[2] == PieceType.S

    def test_dim_background_not_detected(self):
        """Background pixels below threshold should not be detected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        y0, y1 = PREVIEW_SLOTS[0]
        # Fill with dim pixels (below threshold)
        frame[y0:y1, PREVIEW_PIECE_X0:PREVIEW_PIECE_X1] = 30
        preview = read_preview(frame)
        assert preview[0] is None


class TestGhostStripping:
    def test_ghost_removed_from_bottom(self):
        """Ghost cells (dim, at bottom of column) should be stripped."""
        board = np.zeros((20, 10), dtype=bool)
        brightness = np.zeros((20, 10), dtype=float)

        # Real piece at row 17
        board[17, 3:7] = True
        brightness[17, 3:7] = 100  # Bright = real

        # Ghost at row 19 (dim, below real piece)
        board[19, 3:7] = True
        brightness[19, 3:7] = 50  # Dim = ghost

        cleaned = strip_ghost(board, brightness, ghost_max=55)
        # Real cells should remain
        assert cleaned[17, 3:7].all()
        # Ghost cells should be removed
        assert not cleaned[19, 3:7].any()

    def test_bright_bottom_not_stripped(self):
        """Bright cells at bottom should NOT be stripped."""
        board = np.zeros((20, 10), dtype=bool)
        brightness = np.zeros((20, 10), dtype=float)

        board[19, 0:4] = True
        brightness[19, 0:4] = 90  # Bright = real

        cleaned = strip_ghost(board, brightness, ghost_max=55)
        assert cleaned[19, 0:4].all()

    def test_ghost_stops_at_bright_cell(self):
        """Stripping from bottom should stop when hitting a bright cell."""
        board = np.zeros((20, 10), dtype=bool)
        brightness = np.zeros((20, 10), dtype=float)

        # Stack in column 5: bright at 17, dim at 18-19
        board[17, 5] = True
        brightness[17, 5] = 80
        board[18, 5] = True
        brightness[18, 5] = 45  # Ghost
        board[19, 5] = True
        brightness[19, 5] = 50  # Ghost

        cleaned = strip_ghost(board, brightness, ghost_max=55)
        assert cleaned[17, 5]       # Bright cell kept
        assert not cleaned[18, 5]   # Ghost removed
        assert not cleaned[19, 5]   # Ghost removed

    def test_read_board_clean_integration(self):
        """read_board_clean should detect cells and strip ghosts."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Place a bright cell at (5, 15)
        cx, cy = get_cell_center(5, 15)
        x0 = max(0, int(cx - 4))
        x1 = min(640, int(cx + 4))
        y0 = max(0, int(cy - 4))
        y1 = min(480, int(cy + 4))
        frame[y0:y1, x0:x1] = 100  # Bright

        # Place a ghost cell at (5, 19) — dim
        cx2, cy2 = get_cell_center(5, 19)
        x0 = max(0, int(cx2 - 4))
        x1 = min(640, int(cx2 + 4))
        y0 = max(0, int(cy2 - 4))
        y1 = min(480, int(cy2 + 4))
        frame[y0:y1, x0:x1] = 50  # Dim ghost

        board = read_board_clean(frame, threshold=45, ghost_max=55)
        assert board[15, 5]       # Bright cell detected
        assert not board[19, 5]   # Ghost stripped


class TestFallingPieceDetection:
    def test_detect_new_piece(self):
        """Should detect cells present in current but not in settled."""
        settled = np.zeros((20, 10), dtype=bool)
        settled[19, 3:7] = True  # Settled I-piece at bottom

        current = settled.copy()
        current[5, 3:7] = True  # Falling I-piece mid-board

        colors = np.zeros((20, 10, 3), dtype=float)
        colors[5, 3:7] = [40, 135, 190]  # Cyan

        piece_type, cells = detect_falling_piece(current, settled, colors)
        assert piece_type == PieceType.I
        assert len(cells) == 4
        assert all(r == 5 for r, c in cells)

    def test_no_new_piece(self):
        """Should return None when no new cells."""
        board = np.zeros((20, 10), dtype=bool)
        board[19, 3:7] = True
        colors = np.zeros((20, 10, 3), dtype=float)

        piece_type, cells = detect_falling_piece(board, board, colors)
        assert piece_type is None
        assert len(cells) == 0
