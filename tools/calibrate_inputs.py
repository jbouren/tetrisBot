#!/usr/bin/env python3
"""Calibrate input timing and spawn positions for The New Tetris.

Findings so far:
  - Ghost/shadow piece is visible to CV at the landing position
  - 1-frame button holds are too short — only ~33% register
  - Hard drop (U_DPAD) needs >1 frame to register

This script tests different hold durations to find the minimum
that reliably registers button presses.

Usage:
  .venv/bin/python3 tools/calibrate_inputs.py
"""

import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.game.vision import read_board, read_preview, board_to_ascii

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("calibrate")
logging.getLogger("src.emulator.core").setLevel(logging.WARNING)

CORE_LIB = "lib/libmupen64plus.so.2.0.0"
PLUGIN_DIR = "lib"
DATA_DIR = "lib/data"
ROM = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"
GFX = "lib/mupen64plus-video-GLideN64.so"
AUDIO = "lib/mupen64plus-audio-sdl.so"
INPUT = "lib/mupen64plus-input-bot.so"
RSP = "lib/mupen64plus-rsp-hle.so"

FRAME_DELAY = 1 / 120  # small delay, we control frame-by-frame


def press_button(input_server, core, button, hold_frames=2, release_frames=2):
    """Press a button for hold_frames, then release for release_frames."""
    state = ControllerState()
    setattr(state, button, 1)
    for _ in range(hold_frames):
        input_server.set_state(state)
        core.advance_frame()
        time.sleep(FRAME_DELAY)
    input_server.clear()
    for _ in range(release_frames):
        core.advance_frame()
        time.sleep(FRAME_DELAY)


def wait_frames(core, n):
    """Advance n frames with no input."""
    for _ in range(n):
        core.advance_frame()
        time.sleep(FRAME_DELAY)


def reload_and_pause(core, input_server):
    """Reload save state and pause, ready for frame-by-frame control."""
    input_server.clear()
    # Must resume before loading state
    core.resume()
    time.sleep(0.5)
    core.load_state(slot=1)
    time.sleep(2)
    core.pause()
    time.sleep(0.3)
    # Advance a frame so the renderer can produce a screenshot
    core.advance_frame()
    time.sleep(0.1)


def find_occupied_cols(board, row):
    """Return list of occupied column indices in a row."""
    return [c for c in range(board.shape[1]) if board[row, c]]


def main():
    input_server = InputServer(port=8082)
    input_server.start()

    core = Mupen64PlusCore(CORE_LIB, PLUGIN_DIR, DATA_DIR)
    core.startup()
    core.load_rom(ROM)
    core.attach_plugins(gfx=GFX, audio=AUDIO, input=INPUT, rsp=RSP)
    core.execute()

    logger.info("Waiting for emulator to boot...")
    time.sleep(5)

    logger.info("Loading save state (slot 1)...")
    core.load_state(slot=1)
    time.sleep(2)

    results = {}

    try:
        # ── SWEEP: Test different hold durations for hard drop ──
        for hold in [1, 2, 3, 4]:
            logger.info("")
            logger.info("=" * 60)
            logger.info("HARD DROP with %d-frame hold", hold)
            logger.info("=" * 60)

            reload_and_pause(core, input_server)

            frame = core.read_screen()
            board_before = read_board(frame)
            # Find the falling piece row (not row 19 which is ghost)
            falling_row = None
            for r in range(18):  # skip bottom 2 rows (ghost area)
                if board_before[r].any():
                    falling_row = r
                    break
            before_cols = find_occupied_cols(board_before, 19) if board_before[19].any() else []
            logger.info("Before: falling piece at row %s, ghost at row 19 cols %s",
                        falling_row, before_cols)

            press_button(input_server, core, "U_DPAD", hold_frames=hold, release_frames=2)
            wait_frames(core, 20)

            frame = core.read_screen()
            board_after = read_board(frame)

            # Check if piece dropped: falling row should be clear
            if falling_row is not None:
                still_there = board_after[falling_row].any()
                if not still_there:
                    logger.info("SUCCESS! Falling piece at row %d is GONE -> hard drop worked", falling_row)
                    results[f"hard_drop_{hold}f"] = "SUCCESS"
                else:
                    after_cols = find_occupied_cols(board_after, falling_row)
                    logger.info("FAILED: row %d still has cells at cols %s", falling_row, after_cols)
                    results[f"hard_drop_{hold}f"] = "FAILED"
            else:
                logger.warning("Could not find falling piece in before screenshot")
                results[f"hard_drop_{hold}f"] = "UNKNOWN"

            logger.info("After board:\n%s", board_to_ascii(board_after))

        # ── SWEEP: Test different hold durations for LEFT movement ──
        for hold in [1, 2, 3, 4]:
            logger.info("")
            logger.info("=" * 60)
            logger.info("3x LEFT with %d-frame hold each", hold)
            logger.info("=" * 60)

            reload_and_pause(core, input_server)

            frame = core.read_screen()
            board_before = read_board(frame)
            before_cols = find_occupied_cols(board_before, 19)
            before_left = min(before_cols) if before_cols else -1
            logger.info("Before: ghost at row 19, cols %s (leftmost=%d)", before_cols, before_left)

            for i in range(3):
                press_button(input_server, core, "L_DPAD", hold_frames=hold, release_frames=2)

            wait_frames(core, 5)

            frame = core.read_screen()
            board_after = read_board(frame)
            after_cols = find_occupied_cols(board_after, 19)
            after_left = min(after_cols) if after_cols else -1

            moved = before_left - after_left
            logger.info("After: ghost at row 19, cols %s (leftmost=%d)", after_cols, after_left)
            logger.info("Moved %d cells left (expected 3)", moved)
            results[f"left_3x_{hold}f"] = f"moved {moved}/3"

        # ── SWEEP: Test different hold durations for RIGHT movement ──
        for hold in [2, 3]:
            logger.info("")
            logger.info("=" * 60)
            logger.info("3x RIGHT with %d-frame hold each", hold)
            logger.info("=" * 60)

            reload_and_pause(core, input_server)

            frame = core.read_screen()
            board_before = read_board(frame)
            before_cols = find_occupied_cols(board_before, 19)
            before_left = min(before_cols) if before_cols else -1

            for i in range(3):
                press_button(input_server, core, "R_DPAD", hold_frames=hold, release_frames=2)

            wait_frames(core, 5)

            frame = core.read_screen()
            board_after = read_board(frame)
            after_cols = find_occupied_cols(board_after, 19)
            after_left = min(after_cols) if after_cols else -1

            moved = after_left - before_left
            logger.info("After: ghost cols %s (leftmost=%d)", after_cols, after_left)
            logger.info("Moved %d cells right (expected 3)", moved)
            results[f"right_3x_{hold}f"] = f"moved {moved}/3"

        # ── Test rotation ──
        for hold in [2, 3]:
            logger.info("")
            logger.info("=" * 60)
            logger.info("ROTATE CW (A) with %d-frame hold", hold)
            logger.info("=" * 60)

            reload_and_pause(core, input_server)

            frame = core.read_screen()
            board_before = read_board(frame)

            press_button(input_server, core, "A_BUTTON", hold_frames=hold, release_frames=2)
            wait_frames(core, 5)

            frame = core.read_screen()
            board_after = read_board(frame)

            logger.info("BEFORE:                AFTER:")
            before_lines = board_to_ascii(board_before).split("\n")
            after_lines = board_to_ascii(board_after).split("\n")
            for b, a in zip(before_lines, after_lines):
                logger.info("  %s    %s", b, a)

            # I-piece horizontal -> vertical = piece shape changes from #### to single column
            # Check if the ghost piece changed shape
            ghost_before = find_occupied_cols(board_before, 19)
            ghost_after_19 = find_occupied_cols(board_after, 19)
            ghost_after_18 = find_occupied_cols(board_after, 18)
            ghost_after_17 = find_occupied_cols(board_after, 17)
            ghost_after_16 = find_occupied_cols(board_after, 16)

            if len(ghost_before) == 4 and len(ghost_after_19) <= 2:
                logger.info("SUCCESS! I-piece rotated from horizontal (4 wide) to vertical")
                results[f"rotate_{hold}f"] = "SUCCESS"
            else:
                logger.info("Ghost before (row 19): %s  After: row19=%s row18=%s row17=%s row16=%s",
                            ghost_before, ghost_after_19, ghost_after_18, ghost_after_17, ghost_after_16)
                results[f"rotate_{hold}f"] = "check output"

        # ── HOLD-DURATION SWEEP: Hold LEFT for N frames, measure cells moved ──
        # This calibrates the DAS timing: initial move + delay + auto-repeat rate
        logger.info("")
        logger.info("=" * 60)
        logger.info("HOLD-DURATION SWEEP: Hold LEFT for N frames")
        logger.info("=" * 60)

        for hold_frames in [1, 5, 10, 15, 20, 25, 30, 40]:
            reload_and_pause(core, input_server)

            frame = core.read_screen()
            board_before = read_board(frame)
            before_cols = find_occupied_cols(board_before, 19)
            before_left = min(before_cols) if before_cols else -1

            # Hold LEFT for exactly hold_frames
            state = ControllerState(L_DPAD=1)
            for _ in range(hold_frames):
                input_server.set_state(state)
                core.advance_frame()
                time.sleep(FRAME_DELAY)
            input_server.clear()
            wait_frames(core, 3)

            frame = core.read_screen()
            board_after = read_board(frame)
            after_cols = find_occupied_cols(board_after, 19)
            after_left = min(after_cols) if after_cols else -1

            moved = before_left - after_left
            logger.info("  Hold LEFT %2d frames -> moved %d cells (cols %s -> %s)",
                        hold_frames, moved, before_cols, after_cols)
            results[f"hold_left_{hold_frames}f"] = f"moved {moved}"

        # ── Same for RIGHT ──
        logger.info("")
        logger.info("=" * 60)
        logger.info("HOLD-DURATION SWEEP: Hold RIGHT for N frames")
        logger.info("=" * 60)

        for hold_frames in [1, 5, 10, 15, 20, 25, 30, 40]:
            reload_and_pause(core, input_server)

            frame = core.read_screen()
            board_before = read_board(frame)
            before_cols = find_occupied_cols(board_before, 19)
            before_left = min(before_cols) if before_cols else -1

            state = ControllerState(R_DPAD=1)
            for _ in range(hold_frames):
                input_server.set_state(state)
                core.advance_frame()
                time.sleep(FRAME_DELAY)
            input_server.clear()
            wait_frames(core, 3)

            frame = core.read_screen()
            board_after = read_board(frame)
            after_cols = find_occupied_cols(board_after, 19)
            after_left = min(after_cols) if after_cols else -1

            moved = after_left - before_left
            logger.info("  Hold RIGHT %2d frames -> moved %d cells (cols %s -> %s)",
                        hold_frames, moved, before_cols, after_cols)
            results[f"hold_right_{hold_frames}f"] = f"moved {moved}"

    except Exception as e:
        logger.error("Test failed: %s", e, exc_info=True)
    finally:
        input_server.clear()
        try:
            core.resume()
        except Exception:
            pass
        input_server.stop()

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for test, result in results.items():
        logger.info("  %-20s -> %s", test, result)


if __name__ == "__main__":
    main()
