#!/usr/bin/env python3
"""Test brightness-based falling piece detection against captured frames."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from PIL import Image

from src.game.pieces import PIECE_NAMES
from src.game.vision import (
    read_board_colors, read_preview, board_to_ascii,
    detect_falling_piece_by_brightness,
)


def pname(p):
    return PIECE_NAMES[p] if p is not None else "???"


def main():
    print("=== Brightness-Based Falling Piece Detection ===\n")

    prev_preview = None
    for i in range(13):
        path = Path(f"/tmp/tetris_drop_{i:02d}.png")
        if not path.exists():
            break

        frame = np.array(Image.open(path))[:, :, :3]
        occupancy, colors = read_board_colors(frame)
        preview = read_preview(frame)
        piece_type, cells = detect_falling_piece_by_brightness(occupancy, colors)

        # Expected: the falling piece should match preview[0] from the PREVIOUS frame
        expected = pname(prev_preview[0]) if prev_preview else "N/A"

        if cells:
            positions_str = ", ".join(f"({c},{r})" for r, c in cells)
            avg_color = np.mean([colors[r, c] for r, c in cells], axis=0)
            brightness = avg_color.mean()
            match = "OK" if (prev_preview and piece_type == prev_preview[0]) else "  "
            print(f"Frame {i:2d}: falling={pname(piece_type):3s} "
                  f"expected={expected:3s} {match} "
                  f"cells={len(cells)} at {positions_str} "
                  f"br={brightness:.0f} "
                  f"RGB=({avg_color[0]:.0f},{avg_color[1]:.0f},{avg_color[2]:.0f})")
        else:
            print(f"Frame {i:2d}: falling=--- expected={expected:3s}    (not detected)")

        prev_preview = preview

    # Also test game_start and after_drop
    print("\n--- Game start frame ---")
    frame = np.array(Image.open("/tmp/tetris_game_start.png"))[:, :, :3]
    occupancy, colors = read_board_colors(frame)
    piece_type, cells = detect_falling_piece_by_brightness(occupancy, colors)
    if cells:
        positions = ", ".join(f"({c},{r})" for r, c in cells)
        avg = np.mean([colors[r, c] for r, c in cells], axis=0)
        print(f"  Falling: {pname(piece_type)} cells={len(cells)} at {positions} RGB=({avg[0]:.0f},{avg[1]:.0f},{avg[2]:.0f})")

    print("\n--- After drop frame ---")
    frame = np.array(Image.open("/tmp/tetris_after_drop.png"))[:, :, :3]
    occupancy, colors = read_board_colors(frame)
    piece_type, cells = detect_falling_piece_by_brightness(occupancy, colors)
    if cells:
        positions = ", ".join(f"({c},{r})" for r, c in cells)
        avg = np.mean([colors[r, c] for r, c in cells], axis=0)
        print(f"  Falling: {pname(piece_type)} cells={len(cells)} at {positions} RGB=({avg[0]:.0f},{avg[1]:.0f},{avg[2]:.0f})")


if __name__ == "__main__":
    main()
