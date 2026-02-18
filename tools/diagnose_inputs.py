#!/usr/bin/env python3
"""Diagnose input reliability: test rotation and movement on the emulator.

Loads save state, then performs controlled tests:
1. Hard drop with no inputs (baseline)
2. 3x RIGHT tap then hard drop
3. 3x LEFT tap then hard drop
4. 1x rotation then hard drop
Each test reloads the save state for a clean slate.
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.game.vision import read_board, read_board_brightness, board_to_ascii

ROTATE_HOLD = 4
TAP_HOLD = 4
TAP_RELEASE = 3
HARD_DROP_HOLD = 2
LOCK_WAIT = 30

def hold_button(core, input_server, button, frames, frame_delay):
    state = ControllerState()
    setattr(state, button, 1)
    for _ in range(frames):
        input_server.set_state(state)
        core.advance_frame()
        time.sleep(frame_delay)
    input_server.clear()


def wait_frames(core, n, frame_delay):
    for _ in range(n):
        core.advance_frame()
        time.sleep(frame_delay)


def reload_state(core):
    """Reload save state slot 1 and pause."""
    core.resume()
    time.sleep(0.5)
    core.load_state(slot=1)
    time.sleep(2)
    core.pause()
    time.sleep(0.3)
    core.advance_frame()
    time.sleep(0.1)


def read_piece_columns(core, label):
    """Read board and find occupied columns in top 10 rows."""
    frame = core.read_screen()
    board = read_board(frame)
    brightness = read_board_brightness(frame)

    # Find cells in top 10 rows (where the piece should be after drop)
    top_board = board[:10]
    top_bright = brightness[:10]

    occupied = list(zip(*np.where(board)))
    if occupied:
        cols = sorted(set(c for r, c in occupied))
        rows = sorted(set(r for r, c in occupied))
        print(f"  [{label}] Occupied cells: {len(occupied)}")
        print(f"  [{label}] Columns: {cols}")
        print(f"  [{label}] Rows: {rows}")
        # Show specific cells
        for r, c in sorted(occupied):
            print(f"    ({r},{c}) brightness={brightness[r,c]:.0f}")
    else:
        print(f"  [{label}] No occupied cells!")

    print(f"  [{label}] Board:\n{board_to_ascii(board)}")
    return board


def run_test(core, input_server, frame_delay, test_name, actions):
    """Run a single test: reload state, perform actions, hard drop, read board."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    reload_state(core)

    # Wait for piece to spawn
    wait_frames(core, 10, frame_delay)

    # Before screenshot
    print("Before actions:")
    read_piece_columns(core, "before")

    # Perform actions
    for action_name, button, hold_frames in actions:
        print(f"  Action: {action_name} ({button} x {hold_frames}f)")
        hold_button(core, input_server, button, hold_frames, frame_delay)
        wait_frames(core, TAP_RELEASE, frame_delay)

    # After actions, before hard drop
    print("After actions (before drop):")
    read_piece_columns(core, "after_action")

    # Hard drop
    hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD, frame_delay)
    wait_frames(core, LOCK_WAIT, frame_delay)

    # After hard drop
    print("After hard drop:")
    board = read_piece_columns(core, "after_drop")

    return board


def main():
    frame_delay = 1 / 60

    input_server = InputServer(port=8082)
    input_server.start()

    core = Mupen64PlusCore("lib/libmupen64plus.so.2.0.0", "lib", "lib/data")
    core.startup()
    core.load_rom("/mnt/c/code/n64/roms/New Tetris, The (USA).z64")
    core.attach_plugins(
        gfx="lib/mupen64plus-video-GLideN64.so",
        audio="lib/mupen64plus-audio-sdl.so",
        input="lib/mupen64plus-input-bot.so",
        rsp="lib/mupen64plus-rsp-hle.so",
    )
    core.execute()

    print("Waiting for emulator to boot...")
    time.sleep(5)

    # Test 1: Baseline - just hard drop
    run_test(core, input_server, frame_delay, "BASELINE (no inputs, just hard drop)", [])

    # Test 2: 3x RIGHT then hard drop
    run_test(core, input_server, frame_delay, "3x RIGHT (TAP_HOLD=4)", [
        ("RIGHT tap 1", "R_DPAD", TAP_HOLD),
        ("RIGHT tap 2", "R_DPAD", TAP_HOLD),
        ("RIGHT tap 3", "R_DPAD", TAP_HOLD),
    ])

    # Test 3: 3x LEFT then hard drop
    run_test(core, input_server, frame_delay, "3x LEFT (TAP_HOLD=4)", [
        ("LEFT tap 1", "L_DPAD", TAP_HOLD),
        ("LEFT tap 2", "L_DPAD", TAP_HOLD),
        ("LEFT tap 3", "L_DPAD", TAP_HOLD),
    ])

    # Test 4: 1x rotation then hard drop
    run_test(core, input_server, frame_delay, "1x ROTATION (A_BUTTON hold=4)", [
        ("Rotate CW", "A_BUTTON", ROTATE_HOLD),
    ])

    # Test 5: 3x RIGHT with longer hold (TAP_HOLD=8)
    run_test(core, input_server, frame_delay, "3x RIGHT (TAP_HOLD=8)", [
        ("RIGHT tap 1", "R_DPAD", 8),
        ("RIGHT tap 2", "R_DPAD", 8),
        ("RIGHT tap 3", "R_DPAD", 8),
    ])

    # Test 6: 3x LEFT with longer hold (TAP_HOLD=8)
    run_test(core, input_server, frame_delay, "3x LEFT (TAP_HOLD=8)", [
        ("LEFT tap 1", "L_DPAD", 8),
        ("LEFT tap 2", "L_DPAD", 8),
        ("LEFT tap 3", "L_DPAD", 8),
    ])

    core.resume()
    input_server.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
