#!/usr/bin/env python3
"""Analyze captured frames to find board grid coordinates and cell colors.

Uses the saved screenshots to calibrate the CV pipeline.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import cv2
import numpy as np
from PIL import Image


def analyze_frame(path):
    """Analyze a game frame to find board boundaries."""
    img = np.array(Image.open(path))
    print(f"Frame: {path}")
    print(f"  Shape: {img.shape}")

    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # The board area has a distinctive dark background
    # Let's scan for the board by looking at vertical/horizontal brightness profiles

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Horizontal brightness profile (average across rows)
    h_profile = gray.mean(axis=0)
    # Vertical brightness profile (average across columns)
    v_profile = gray.mean(axis=1)

    print(f"\n  Horizontal profile (brightness by column):")
    # Find dark regions (board background is dark)
    # Sample every 20 pixels
    for x in range(0, img.shape[1], 20):
        bar = "#" * int(h_profile[x] / 5)
        print(f"    x={x:3d}: {h_profile[x]:5.1f} {bar}")

    print(f"\n  Vertical profile (brightness by row):")
    for y in range(0, img.shape[0], 20):
        bar = "#" * int(v_profile[y] / 5)
        print(f"    y={y:3d}: {v_profile[y]:5.1f} {bar}")

    # Look for the board region: it should be a rectangular area that's
    # darker than the ornate borders but has colored blocks inside.
    # Let's scan for columns that have very dark pixels (board background)
    # The board background in The New Tetris is very dark blue/black

    # Check which columns have many dark pixels (value < 30)
    dark_mask = gray < 30
    dark_cols = dark_mask.sum(axis=0)
    dark_rows = dark_mask.sum(axis=1)

    print(f"\n  Dark pixel count by column (every 10px):")
    for x in range(0, img.shape[1], 10):
        bar = "#" * (dark_cols[x] // 10)
        print(f"    x={x:3d}: {dark_cols[x]:3d} {bar}")

    print(f"\n  Dark pixel count by row (every 10px):")
    for y in range(0, img.shape[0], 10):
        bar = "#" * (dark_rows[y] // 10)
        print(f"    y={y:3d}: {dark_rows[y]:3d} {bar}")

    # Now let's try to find the exact board edges.
    # The board is surrounded by ornate stone borders.
    # Inside the board, empty cells are very dark.
    # Let's look for a rectangular region where columns have high dark-pixel counts.

    # Find contiguous dark column region
    threshold = img.shape[0] * 0.3  # At least 30% of column is dark
    board_cols = dark_cols > threshold
    col_starts = []
    col_ends = []
    in_region = False
    for x in range(img.shape[1]):
        if board_cols[x] and not in_region:
            col_starts.append(x)
            in_region = True
        elif not board_cols[x] and in_region:
            col_ends.append(x)
            in_region = False
    if in_region:
        col_ends.append(img.shape[1])

    print(f"\n  Dark column regions (>30% dark):")
    for s, e in zip(col_starts, col_ends):
        print(f"    x={s} to x={e} (width={e-s})")

    # Similarly for rows
    threshold_r = img.shape[1] * 0.1
    board_rows = dark_rows > threshold_r
    row_starts = []
    row_ends = []
    in_region = False
    for y in range(img.shape[0]):
        if board_rows[y] and not in_region:
            row_starts.append(y)
            in_region = True
        elif not board_rows[y] and in_region:
            row_ends.append(y)
            in_region = False
    if in_region:
        row_ends.append(img.shape[0])

    print(f"\n  Dark row regions (>10% dark):")
    for s, e in zip(row_starts, row_ends):
        print(f"    y={s} to y={e} (height={e-s})")

    # Now let's manually sample specific rows/columns to find the grid lines
    # The board in The New Tetris has 10 columns and 20 rows
    # Let's sample a specific horizontal slice to find column boundaries

    # Use a row in the middle of the board area (around y=300 based on visual)
    print(f"\n  --- Sampling horizontal line at y=350 ---")
    row_pixels = img[350, :, :]
    row_gray = gray[350, :]
    for x in range(0, img.shape[1], 5):
        r, g, b = row_pixels[x]
        v = row_gray[x]
        marker = " " if v < 20 else "*" if v < 50 else "#"
        if 100 <= x <= 400:  # Focus on likely board area
            print(f"    x={x:3d}: RGB=({r:3d},{g:3d},{b:3d}) gray={v:3d} {marker}")

    # Sample vertical line at x=250 (middle of board)
    print(f"\n  --- Sampling vertical line at x=200 ---")
    col_pixels = img[:, 200, :]
    col_gray = gray[:, 200]
    for y in range(0, img.shape[0], 5):
        r, g, b = col_pixels[y]
        v = col_gray[y]
        if 50 <= y <= 450:
            marker = " " if v < 20 else "*" if v < 50 else "#"
            print(f"    y={y:3d}: RGB=({r:3d},{g:3d},{b:3d}) gray={v:3d} {marker}")

    # Let's also look at the "after drop" image to see placed blocks
    return img


def find_grid_edges(img):
    """Try to find the precise board grid from pixel data."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # The board in The New Tetris appears to occupy roughly:
    # - Left edge around x=130-140
    # - Right edge around x=350-360
    # - Top edge around y=35-45
    # - Bottom edge around y=420-430
    # Let's verify by scanning for transitions

    # Scan horizontal at y=300 (mid-board) for left and right edges
    row = gray[300, :]
    # Left edge: first transition from bright to dark
    for x in range(50, 300):
        if row[x] < 15 and row[x-3] > 30:
            print(f"\n  Left edge candidate: x={x} (gray {row[x-3]} -> {row[x]})")
            break

    # Right edge: last transition from dark to bright
    for x in range(400, 200, -1):
        if row[x] > 30 and row[x-3] < 15:
            print(f"  Right edge candidate: x={x} (gray {row[x-3]} -> {row[x]})")
            break

    # Scan vertical at x=200 for top and bottom
    col = gray[:, 200]
    for y in range(20, 200):
        if col[y] < 15 and col[y-3] > 30:
            print(f"  Top edge candidate: y={y} (gray {col[y-3]} -> {col[y]})")
            break

    for y in range(460, 200, -1):
        if col[y] > 30 and col[y-3] < 15:
            print(f"  Bottom edge candidate: y={y} (gray {col[y-3]} -> {col[y]})")
            break

    # Let's also look at color samples in the placed blocks area
    print(f"\n  --- Color samples of placed blocks (bottom of board) ---")
    # From the after_drop image, blocks should be around y=380-420
    for y in range(360, 430, 10):
        for x in range(140, 370, 20):
            r, g, b = img[y, x]
            v = gray[y, x]
            if v > 30:  # non-dark = block
                print(f"    ({x:3d},{y:3d}): RGB=({r:3d},{g:3d},{b:3d}) gray={v:3d}")


def main():
    print("=" * 60)
    print("Grid Calibration Tool")
    print("=" * 60)

    game_start = Path("/tmp/tetris_game_start.png")
    after_drop = Path("/tmp/tetris_after_drop.png")

    if not game_start.exists():
        print("Run capture_test.py first to generate screenshots!")
        return

    print("\n>>> Analyzing game start frame:")
    img1 = analyze_frame(game_start)
    find_grid_edges(img1)

    if after_drop.exists():
        print("\n\n>>> Analyzing after-drop frame:")
        img2 = analyze_frame(after_drop)
        find_grid_edges(img2)

        # Diff between frames: where did blocks appear?
        diff = np.abs(img2.astype(int) - img1.astype(int)).astype(np.uint8)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        changed = diff_gray > 30
        print(f"\n  Pixels that changed significantly: {changed.sum()}")

        # Find bounding box of changes
        ys, xs = np.where(changed)
        if len(ys) > 0:
            print(f"  Change bounding box: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
