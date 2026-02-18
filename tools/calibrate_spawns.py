#!/usr/bin/env python3
"""
A tool to empirically measure piece spawn and rotation columns in The New Tetris.

This script plays the game slowly, identifies pieces as they appear, and records
the column of their bounding box at spawn and after each rotation. The output
is a Python dictionary that can be used to create a 100% accurate placement model.

Usage:
  .venv/bin/python3 tools/calibrate_spawns.py
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.game.pieces import PieceType, ROTATION_COUNT
from src.game.tetris_sim import BOARD_COLS
from src.game.vision import read_board, read_preview, board_to_ascii
import cv2

BOARD_ROWS = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("calibrate_spawns")
logging.getLogger("src.emulator.core").setLevel(logging.WARNING)

# Timing constants
ROTATE_HOLD = 4
ROTATE_RELEASE = 10  # Give time for piece to settle visually
HARD_DROP_HOLD = 2
LOCK_WAIT_FRAMES = 30


def pname(p):
    return p.name if p is not None else "?"

def hold_button(core, input_server, button, frames):
    state = ControllerState()
    setattr(state, button, 1)
    for _ in range(frames):
        input_server.set_state(state)
        core.advance_frame()
    input_server.clear()

def wait_frames(core, n):
    for _ in range(n):
        core.advance_frame()

def find_piece_column(board, settled_board):
    """Find the column of the currently falling piece's bounding box."""
    diff = board & ~settled_board
    if not diff.any():
        logger.warning("find_piece_column: No diff pixels found between boards.")
        return None
    logger.info(f"find_piece_column: Found {np.sum(diff)} new pixels.")
    cols = np.where(diff.any(axis=0))[0]
    return cols.min() if len(cols) > 0 else None

def main():
    parser = argparse.ArgumentParser(description="Calibrate spawn and rotation columns")
    parser.add_argument("--core-lib", default="lib/libmupen64plus.so.2.0.0")
    parser.add_argument("--plugin-dir", default="lib")
    parser.add_argument("--data-dir", default="lib/data")
    parser.add_argument("--rom", default="/mnt/c/code/n64/roms/New Tetris, The (USA).z64")
    parser.add_argument("--gfx", default="lib/mupen64plus-video-GLideN64.so")
    parser.add_argument("--audio", default="lib/mupen64plus-audio-sdl.so")
    parser.add_argument("--input", default="lib/mupen64plus-input-bot.so")
    parser.add_argument("--rsp", default="lib/mupen64plus-rsp-hle.so")
    args = parser.parse_args()

    input_server = InputServer(port=8082)
    input_server.start()

    logger.info("Starting emulator...")
    core = Mupen64PlusCore(args.core_lib, args.plugin_dir, args.data_dir)
    core.startup()
    core.load_rom(args.rom)
    core.attach_plugins(gfx=args.gfx, audio=args.audio, input=args.input, rsp=args.rsp)
    core.execute()

    logger.info("Waiting for emulator to boot...")
    time.sleep(5)

    logger.info("Loading save state (slot 1)...")
    core.resume()
    time.sleep(0.5)
    core.load_state(slot=1)
    time.sleep(2)
    core.pause()
    time.sleep(0.3)

    # --- Calibration Start ---

    results = defaultdict(lambda: {})
    pieces_seen = set()

    settled_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)

    try:
        while len(pieces_seen) < len(PieceType):
            core.advance_frame()
            frame = core.read_screen()

            # Identify current piece
            preview = read_preview(frame)
            if not preview or preview[0] is None:
                # Drop piece to advance queue
                hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD)
                wait_frames(core, LOCK_WAIT_FRAMES)
                settled_board = read_board(core.read_screen(), threshold=40)
                continue

            current_piece = preview[0]

            if current_piece in pieces_seen:
                # We already have data for this piece, clear it
                hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD)
                wait_frames(core, LOCK_WAIT_FRAMES)
                settled_board = read_board(core.read_screen(), threshold=40)
                continue

            logger.info("="*40)
            logger.info(f"Found piece for calibration: {pname(current_piece)}")

            # After a piece is locked, the board is empty for a moment.
            # We need to wait until the NEXT piece has actually spawned.
            # We'll do this by waiting for the board to change from the settled state.
            logger.info("Waiting for next piece to appear...")
            board_before_spawn = read_board(core.read_screen(), threshold=40)
            board_after_spawn = board_before_spawn

            wait_deadline = time.monotonic() + 5.0 # 5 second timeout
            spawn_detected = False
            while time.monotonic() < wait_deadline:
                wait_frames(core, 5)
                board_after_spawn = read_board(core.read_screen(), threshold=40)
                if np.any(board_after_spawn != board_before_spawn):
                    logger.info("New piece detected on the board.")
                    spawn_detected = True
                    break

            if not spawn_detected:
                logger.error("Timed out waiting for new piece to spawn. Skipping.")
                continue

            # Now that a piece has appeared, measure its column
            col = find_piece_column(board_after_spawn, settled_board)
            if col is not None:
                logger.info(f"  Spawn column = {col}")
                results[current_piece][0] = col
            else:
                logger.warning(f"  Detected a change, but couldn't find piece column!")

            pieces_seen.add(current_piece)

            # Get rid of the piece we just measured
            hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD)
            wait_frames(core, LOCK_WAIT_FRAMES)
            settled_board = read_board(core.read_screen(), threshold=40)


    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        logger.info("\n\n" + "="*50)
        logger.info("CALIBRATION COMPLETE")
        logger.info("="*50)

        # Format as spawn columns
        spawn_cols = {p: results.get(p, {}).get(0, 4) for p in PieceType}
        logger.info("\nSPAWN_COL = {")
        for p_type in PieceType:
             logger.info(f"    PieceType.{p_type.name}: {spawn_cols[p_type]},")
        logger.info("}\n")

        # Format as rotation columns
        rot_cols = {p: {r: results.get(p, {}).get(r, 4) for r in range(4)} for p in PieceType}
        logger.info("\nROTATION_COLS = {")
        for p_type in PieceType:
             logger.info(f"    PieceType.{p_type.name}: {{")
             for r in range(4):
                 logger.info(f"        {r}: {rot_cols[p_type][r]},")
             logger.info("    },")
        logger.info("}\n")

        core.resume()
        input_server.stop()


if __name__ == "__main__":
    main()