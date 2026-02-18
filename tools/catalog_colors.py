#!/usr/bin/env python3
"""Catalog piece colors by capturing many frames.

Drops pieces repeatedly and records the color of each preview piece
and each falling piece on the board, building a color-to-piece-type map.
"""

import logging
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from PIL import Image

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.game.vision import read_board_colors, board_to_ascii

if "GALLIUM_DRIVER" not in os.environ:
    os.environ["GALLIUM_DRIVER"] = "d3d12"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

LIB_DIR = PROJECT_DIR / "lib"
DATA_DIR = LIB_DIR / "data"
ROM_PATH = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"

# Preview slot regions (approximate centers and sampling areas)
# Based on analysis: pieces at x≈316-358, in 3 vertical slots
PREVIEW_X0 = 316
PREVIEW_X1 = 358
PREVIEW_SLOTS = [
    (66, 88),    # Slot 1: y range
    (118, 142),  # Slot 2: y range
    (174, 198),  # Slot 3: y range
]


def sample_preview_slot(frame, slot_idx):
    """Sample the average color of bright pixels in a preview slot."""
    y0, y1 = PREVIEW_SLOTS[slot_idx]
    region = frame[y0:y1, PREVIEW_X0:PREVIEW_X1]

    # Only average the bright pixels (the piece itself, not background)
    brightness = region.mean(axis=2)
    bright_mask = brightness > 60

    if bright_mask.sum() < 5:
        return None  # No piece visible

    bright_pixels = region[bright_mask]
    return bright_pixels.mean(axis=0)


def color_signature(rgb):
    """Create a compact color signature string."""
    if rgb is None:
        return "---"
    r, g, b = rgb
    return f"({r:.0f},{g:.0f},{b:.0f})"


def classify_color(rgb):
    """Rough initial classification of a color."""
    if rgb is None:
        return "empty"
    r, g, b = rgb

    # Normalize
    total = r + g + b
    if total < 50:
        return "dark"

    # Check for gray/white (all channels similar)
    if max(r, g, b) - min(r, g, b) < 30:
        return "gray/white"

    # Blue dominant
    if b > r and b > g:
        if g > r:
            return "cyan"
        return "blue"

    # Red dominant
    if r > g and r > b:
        if g > b and g > r * 0.6:
            return "yellow/orange"
        if g < r * 0.3:
            return "red"
        return "orange"

    # Green dominant
    if g > r and g > b:
        if r > g * 0.7:
            return "yellow"
        return "green"

    return f"?({r:.0f},{g:.0f},{b:.0f})"


def main():
    print("=== Piece Color Catalog ===", flush=True)

    input_server = InputServer(host="127.0.0.1", port=8082)
    input_server.start()

    core = Mupen64PlusCore(
        core_lib_path=str(LIB_DIR / "libmupen64plus.so.2.0.0"),
        plugin_dir=str(LIB_DIR),
        data_dir=str(DATA_DIR),
    )

    core.startup()
    core.load_rom(ROM_PATH)
    core.attach_plugins(
        gfx=str(LIB_DIR / "mupen64plus-video-GLideN64.so"),
        audio=None,
        input=str(LIB_DIR / "mupen64plus-input-bot.so"),
        rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    )

    print("Starting emulation...", flush=True)
    core.execute()
    time.sleep(2)

    print("Loading save state...", flush=True)
    core.load_state(slot=1)
    time.sleep(1)
    core.pause()
    time.sleep(0.2)

    prev_board = None
    all_colors = []

    for drop_num in range(20):
        print(f"\n--- Drop {drop_num} ---", flush=True)

        # Capture frame before drop
        frame = core.read_screen()

        # Read board
        occupancy, colors = read_board_colors(frame)

        # Find falling piece cells (cells not in previous settled board)
        if prev_board is not None:
            new_cells = occupancy & ~prev_board
            new_positions = list(zip(*np.where(new_cells)))
            if new_positions:
                # Get color of falling piece
                piece_colors = [colors[r, c] for r, c in new_positions]
                avg_color = np.mean(piece_colors, axis=0)
                classification = classify_color(avg_color)
                positions_str = ", ".join(f"({c},{r})" for r, c in new_positions)
                print(f"  Falling piece: {len(new_positions)} cells at {positions_str}")
                print(f"    Color: {color_signature(avg_color)} -> {classification}")
            else:
                print("  No new cells detected (piece may be above board)")

        # Read preview slots
        print("  Preview slots:", flush=True)
        for slot in range(3):
            rgb = sample_preview_slot(frame, slot)
            cls = classify_color(rgb)
            print(f"    Slot {slot+1}: {color_signature(rgb)} -> {cls}")
            all_colors.append({"drop": drop_num, "slot": slot, "rgb": rgb, "cls": cls})

        # Print board
        print(board_to_ascii(occupancy))

        # Save frame
        img = Image.fromarray(frame, "RGB")
        img.save(f"/tmp/tetris_drop_{drop_num:02d}.png")

        # Hard drop the piece
        core.resume()
        time.sleep(0.05)
        state = ControllerState()
        state.U_DPAD = 1
        input_server.set_state(state)
        time.sleep(0.05)
        input_server.clear()
        time.sleep(1.5)
        core.pause()
        time.sleep(0.2)

        # Capture settled board after drop
        frame_after = core.read_screen()
        prev_board, _ = read_board_colors(frame_after)

    # Summary
    print("\n\n=== COLOR SUMMARY ===")
    print("All unique color classifications seen:")
    seen = {}
    for entry in all_colors:
        cls = entry["cls"]
        rgb = entry["rgb"]
        if cls not in seen and rgb is not None:
            seen[cls] = color_signature(rgb)
    for cls, sig in sorted(seen.items()):
        print(f"  {cls:20s} -> {sig}")

    print(f"\nTotal drops: 20, Total preview observations: {len(all_colors)}")
    print("\nDone!", flush=True)

    try:
        core.stop()
    except Exception:
        pass
    input_server.stop()


if __name__ == "__main__":
    main()
