#!/usr/bin/env python3
"""Test piece detection against captured frames."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from PIL import Image

from src.game.pieces import PieceType, PIECE_NAMES
from src.game.vision import (
    read_board, read_board_colors, read_preview, board_to_ascii,
    detect_falling_piece, classify_color, sample_preview_slot,
)


def piece_name(p):
    if p is None:
        return "???"
    return PIECE_NAMES[p]


def test_offline_frames():
    """Test detection using saved frames from catalog_colors.py."""
    print("=== Offline Detection Test ===\n")

    prev_settled = None
    for i in range(20):
        path = Path(f"/tmp/tetris_drop_{i:02d}.png")
        if not path.exists():
            break

        frame = np.array(Image.open(path))[:, :, :3]
        occupancy, colors = read_board_colors(frame)
        preview = read_preview(frame)

        print(f"--- Frame {i} ---")
        print(f"  Preview: [{piece_name(preview[0])}, {piece_name(preview[1])}, {piece_name(preview[2])}]")

        # Show preview raw colors for verification
        for slot in range(3):
            rgb = sample_preview_slot(frame, slot)
            if rgb is not None:
                print(f"    Slot {slot}: RGB=({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f}) -> {piece_name(preview[slot])}")

        if prev_settled is not None:
            piece_type, cells = detect_falling_piece(occupancy, prev_settled, colors)
            if cells:
                positions_str = ", ".join(f"({c},{r})" for r, c in cells)
                print(f"  Falling: {piece_name(piece_type)} at {positions_str}")
            else:
                print(f"  Falling: (not detected)")

        print(board_to_ascii(occupancy))

        # Use this frame's board as the "settled" board for next iteration
        # (This is approximate - in reality we'd capture after the piece lands)
        prev_settled = occupancy

    print("\nDone!")


def test_game_start():
    """Test with the game start and after-drop frames."""
    print("=== Game Start / After Drop Test ===\n")

    start_path = Path("/tmp/tetris_game_start.png")
    drop_path = Path("/tmp/tetris_after_drop.png")

    if start_path.exists():
        frame = np.array(Image.open(start_path))[:, :, :3]
        preview = read_preview(frame)
        print(f"Game start preview: [{piece_name(preview[0])}, {piece_name(preview[1])}, {piece_name(preview[2])}]")

        occupancy, colors = read_board_colors(frame)
        print("Board:")
        print(board_to_ascii(occupancy))

        # The falling piece here is the I-piece (#### at row 7)
        # The settled board has just one row at the bottom
        empty = np.zeros_like(occupancy)
        # Row 19 has settled blocks
        settled = np.zeros_like(occupancy)
        settled[19, 3:7] = True  # The I piece from a previous drop
        piece_type, cells = detect_falling_piece(occupancy, settled, colors)
        print(f"Falling piece (vs empty+row19): {piece_name(piece_type)}")
        if cells:
            for r, c in cells:
                print(f"  ({c},{r}): RGB=({colors[r,c,0]:.0f},{colors[r,c,1]:.0f},{colors[r,c,2]:.0f})")

    if drop_path.exists():
        frame = np.array(Image.open(drop_path))[:, :, :3]
        preview = read_preview(frame)
        print(f"\nAfter drop preview: [{piece_name(preview[0])}, {piece_name(preview[1])}, {piece_name(preview[2])}]")


if __name__ == "__main__":
    # Add color_signature if not in vision.py yet
    if not hasattr(sys.modules.get('src.game.vision', None), 'color_signature'):
        def color_signature(rgb):
            if rgb is None: return "---"
            return f"({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f})"

    test_game_start()
    print()
    test_offline_frames()
