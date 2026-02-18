#!/usr/bin/env python3
"""Live test of piece detection with proper settled-board tracking.

Loads save state, then for each piece:
1. Capture frame → read preview + detect falling piece
2. Hard drop the piece
3. Capture settled board (no falling piece)
4. Repeat
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
from src.game.pieces import PIECE_NAMES
from src.game.vision import (
    read_board_colors, read_preview, board_to_ascii,
    detect_falling_piece,
)

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


def pname(p):
    return PIECE_NAMES[p] if p is not None else "???"


def main():
    print("=== Live Detection Test ===", flush=True)

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

    # Capture initial settled board (save state is at game start with 1 settled I-piece)
    # We need to first hard drop the initial falling piece to get a clean settled state
    print("\nDropping initial piece to get clean settled board...", flush=True)
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

    # Now capture the settled board (piece has landed, next hasn't appeared yet—or just appeared)
    # Advance a few frames to ensure piece is settled and new piece appears
    for _ in range(30):
        core.advance_frame()
    time.sleep(0.1)

    frame = core.read_screen()
    settled_board, _ = read_board_colors(frame)
    print("Settled board after initial drop:")
    print(board_to_ascii(settled_board))

    # Now track pieces
    results = []
    for drop_num in range(15):
        print(f"\n{'='*40}")
        print(f"Piece #{drop_num + 1}", flush=True)

        # Wait for new piece to appear and start falling
        # Advance some frames
        for _ in range(60):
            core.advance_frame()
        time.sleep(0.1)

        # Capture frame with falling piece
        frame = core.read_screen()
        occupancy, colors = read_board_colors(frame)
        preview = read_preview(frame)

        # Detect falling piece
        piece_type, cells = detect_falling_piece(occupancy, settled_board, colors)

        preview_str = f"[{pname(preview[0])}, {pname(preview[1])}, {pname(preview[2])}]"
        print(f"  Preview: {preview_str}")

        if cells:
            positions_str = ", ".join(f"({c},{r})" for r, c in cells)
            avg_color = np.mean([colors[r, c] for r, c in cells], axis=0)
            print(f"  Falling: {pname(piece_type)} ({len(cells)} cells) at {positions_str}")
            print(f"    Color: ({avg_color[0]:.0f},{avg_color[1]:.0f},{avg_color[2]:.0f})")
        else:
            print(f"  Falling: not detected (piece may be at top)")

        print(board_to_ascii(occupancy))

        results.append({
            "drop": drop_num,
            "preview": [preview[0], preview[1], preview[2]],
            "falling": piece_type,
            "cells": len(cells),
        })

        # Hard drop
        core.resume()
        time.sleep(0.05)
        state = ControllerState()
        state.U_DPAD = 1
        input_server.set_state(state)
        time.sleep(0.05)
        input_server.clear()
        time.sleep(1.0)
        core.pause()
        time.sleep(0.2)

        # Advance frames for piece to settle
        for _ in range(30):
            core.advance_frame()
        time.sleep(0.1)

        # Capture new settled board
        frame = core.read_screen()
        settled_board, _ = read_board_colors(frame)

    # Summary
    print(f"\n\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    for r in results:
        prev = f"[{pname(r['preview'][0])},{pname(r['preview'][1])},{pname(r['preview'][2])}]"
        print(f"  Piece {r['drop']+1:2d}: falling={pname(r['falling']):3s} cells={r['cells']} preview={prev}")

    # Check consistency: falling piece should match what was in preview[0] of previous frame
    print("\n  Consistency check (falling == prev frame's preview[0]):")
    for i in range(1, len(results)):
        prev_next = results[i - 1]["preview"][0]
        curr_fall = results[i]["falling"]
        match = "✓" if prev_next == curr_fall else "✗"
        print(f"    Piece {i+1}: preview[0]={pname(prev_next)} falling={pname(curr_fall)} {match}")

    print("\nDone!", flush=True)
    try:
        core.stop()
    except Exception:
        pass
    input_server.stop()


if __name__ == "__main__":
    main()
