#!/usr/bin/env python3
"""Diagnose ghost piece brightness to calibrate CV thresholds.

Takes a screenshot and shows the brightness of every cell, highlighting
different brightness bands:
  - Dark (<20): empty
  - Low (20-40): might be ghost
  - Medium (40-65): likely ghost
  - High (65-85): settled piece (dim)
  - Bright (>85): settled piece or falling piece

Usage:
  .venv/bin/python3 tools/diagnose_ghost.py [screenshot.png]

If no screenshot is provided, captures one from the running emulator.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.game.vision import (
    read_board_brightness, read_board, BOARD_ROWS, BOARD_COLS,
)


def show_brightness(brightness):
    """Print color-coded brightness map."""
    print("\nBrightness map (each cell shows brightness value):")
    print("    " + "".join(f" {c:>3}" for c in range(BOARD_COLS)))
    print("    " + "-" * 40)
    for row in range(BOARD_ROWS):
        vals = ""
        for col in range(BOARD_COLS):
            b = brightness[row, col]
            vals += f" {b:3.0f}"
        print(f" {row:2d}|{vals}")

    print("\nHistogram of non-dark cells (brightness > 10):")
    bright_cells = brightness[brightness > 10].flatten()
    if len(bright_cells) == 0:
        print("  (no bright cells)")
        return

    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 256]
    counts, edges = np.histogram(bright_cells, bins=bins)
    for i, count in enumerate(counts):
        if count > 0:
            bar = "#" * count
            print(f"  {edges[i]:3.0f}-{edges[i+1]:3.0f}: {count:3d} {bar}")

    print(f"\n  Total non-dark cells: {len(bright_cells)}")
    print(f"  Min: {bright_cells.min():.1f}  Max: {bright_cells.max():.1f}  Mean: {bright_cells.mean():.1f}")

    # Show what different thresholds would capture
    for t in [20, 40, 50, 60, 65, 70, 75, 80]:
        n = (brightness > t).sum()
        print(f"  Threshold {t}: {n} cells occupied")


def main():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        # Load from file
        import cv2
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print(f"Failed to load {sys.argv[1]}")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"Loaded {sys.argv[1]} ({frame.shape})")
    else:
        # Capture from emulator
        from src.emulator.core import Mupen64PlusCore
        import time

        core_lib = "lib/libmupen64plus.so.2.0.0"
        core = Mupen64PlusCore(core_lib, "lib", "lib/data")
        core.startup()
        core.load_rom("/mnt/c/code/n64/roms/New Tetris, The (USA).z64")
        core.attach_plugins(
            gfx="lib/mupen64plus-video-GLideN64.so",
            audio="lib/mupen64plus-audio-sdl.so",
            input="lib/mupen64plus-input-bot.so",
            rsp="lib/mupen64plus-rsp-hle.so",
        )
        core.execute()
        time.sleep(5)
        core.resume()
        time.sleep(0.5)
        core.load_state(slot=1)
        time.sleep(2)
        core.pause()
        time.sleep(0.3)
        core.advance_frame()

        frame = core.read_screen()
        core.resume()
        print("Captured screenshot from emulator")

    brightness = read_board_brightness(frame)
    show_brightness(brightness)

    # Show board at different thresholds
    for t in [20, 40, 65, 75]:
        board = read_board(frame, threshold=t)
        from src.game.vision import board_to_ascii
        print(f"\nBoard at threshold={t}:")
        print(board_to_ascii(board))


if __name__ == "__main__":
    main()
