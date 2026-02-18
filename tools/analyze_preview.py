#!/usr/bin/env python3
"""Analyze the preview box area to understand piece colors and layout."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from PIL import Image

PREVIEW_LEFT = 310
PREVIEW_TOP = 40
PREVIEW_RIGHT = 360
PREVIEW_BOTTOM = 200

def analyze_preview(path):
    img = np.array(Image.open(path))
    print(f"\nFrame: {path.name}")
    print(f"Preview box: ({PREVIEW_LEFT},{PREVIEW_TOP}) to ({PREVIEW_RIGHT},{PREVIEW_BOTTOM})")
    print(f"Size: {PREVIEW_RIGHT - PREVIEW_LEFT}x{PREVIEW_BOTTOM - PREVIEW_TOP}")

    preview = img[PREVIEW_TOP:PREVIEW_BOTTOM, PREVIEW_LEFT:PREVIEW_RIGHT]

    # Show brightness profile vertically (to find piece slot boundaries)
    print("\nVertical brightness profile (avg brightness per row):")
    for y in range(0, PREVIEW_BOTTOM - PREVIEW_TOP, 2):
        row_brightness = preview[y].mean()
        bar = "#" * int(row_brightness / 4)
        print(f"  y={PREVIEW_TOP + y:3d}: {row_brightness:5.1f} {bar}")

    # Show horizontal brightness at a few vertical slices
    print("\nHorizontal brightness profiles:")
    for rel_y in [20, 40, 60, 80, 100, 120, 140]:
        abs_y = PREVIEW_TOP + rel_y
        if abs_y >= PREVIEW_BOTTOM:
            break
        print(f"  --- y={abs_y} ---")
        for x in range(PREVIEW_LEFT, PREVIEW_RIGHT, 3):
            r, g, b = img[abs_y, x]
            brightness = (int(r) + int(g) + int(b)) // 3
            marker = "." if brightness < 20 else "*" if brightness < 50 else "#"
            print(f"    x={x:3d}: RGB=({r:3d},{g:3d},{b:3d}) br={brightness:3d} {marker}")

    # Sample a grid of the preview area
    print("\nPreview area color grid (every 5px):")
    header = "     " + "".join(f"{x:4d}" for x in range(PREVIEW_LEFT, PREVIEW_RIGHT, 5))
    print(header)
    for y in range(PREVIEW_TOP, PREVIEW_BOTTOM, 5):
        row_str = f"y={y:3d} "
        for x in range(PREVIEW_LEFT, PREVIEW_RIGHT, 5):
            r, g, b = img[y, x]
            brightness = (int(r) + int(g) + int(b)) // 3
            if brightness < 15:
                row_str += "   ."
            elif brightness < 40:
                row_str += "   -"
            else:
                # Show dominant color channel
                if r > g and r > b:
                    row_str += "   R"
                elif g > r and g > b:
                    row_str += "   G"
                elif b > r and b > g:
                    row_str += "   B"
                else:
                    row_str += "   W"

        print(row_str)

    # Also analyze the current piece on the board by looking at color clusters
    print("\n\n=== Board color analysis ===")
    from src.game.vision import read_board_colors, BOARD_ROWS, BOARD_COLS
    occupancy, colors = read_board_colors(img)

    print("Occupied cells with colors:")
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if occupancy[row, col]:
                r, g, b = colors[row, col]
                print(f"  ({col},{row}): RGB=({r:.0f},{g:.0f},{b:.0f})")


def main():
    for path in [Path("/tmp/tetris_game_start.png"), Path("/tmp/tetris_after_drop.png")]:
        if path.exists():
            analyze_preview(path)
            print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
